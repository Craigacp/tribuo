/*
 * Copyright (c) 2023, Oracle and/or its affiliates. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tribuo.util.sentencepiece;

import org.tribuo.util.sentencepiece.protos.SentencepieceModel;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;

public abstract sealed class SPModel permits BPESPModel, CharSPModel, WordSPModel, UnigramSPModel {

    protected static final HexFormat HEX_FORMATTER = HexFormat.of().withPrefix("<0x").withSuffix(">").withUpperCase();

    public enum ExtraOptions {REVERSE, ADD_BOS, ADD_EOS, UNK}

    public static final String DEFAULT_UNKNOWN = "<unk>";
    public static final String DEFAULT_BOS = "<s>";
    public static final String DEFAULT_EOS = "</s>";
    public static final String DEFAULT_PAD = "<pad>";
    public static final int DEFAULT_UNKNOWN_ID = 0;
    public static final int DEFAULT_BOS_ID = 1;
    public static final int DEFAULT_EOS_ID = 2;
    public static final int DEFAULT_PAD_ID = -1;

    protected final Map<String, Integer> vocab;
    protected final String[] inverseVocab;
    // This array is ragged.
    protected final byte[][] inverseVocabUTF8;

    protected final Map<String, Integer> reservedIdMap;

    protected final Trie userDefinedSymbolTrie;

    protected final int unkId;
    protected final int bosId;
    protected final int eosId;
    protected final int padId;

    protected final String unk;
    protected final String bos;
    protected final String eos;
    protected final String pad;
    protected final boolean byteFallback;

    protected final EnumSet<ExtraOptions> options;

    protected final Normalizer normalizer;
    protected final Normalizer denormalizer;

    protected final SentencepieceModel.ModelProto proto;

    protected SPModel(SentencepieceModel.ModelProto proto, EnumSet<ExtraOptions> options) {
        Set<String> userDefinedSymbols = new HashSet<>();
        boolean[] bytes = new boolean[256];
        this.vocab = new HashMap<>(proto.getPiecesCount());
        this.reservedIdMap = new HashMap<>();
        int tmpUnk = -1;
        for (int i = 0; i < proto.getPiecesCount(); i++) {
            var piece = proto.getPieces(i);
            var type = piece.getType();
            var output = switch (type) {
                case NORMAL, USER_DEFINED, UNUSED -> vocab.put(piece.getPiece(), i);
                case UNKNOWN, CONTROL, BYTE -> reservedIdMap.put(piece.getPiece(), i);
            };
            if (output != null) {
                throw new IllegalStateException("Found a duplicate mapping for piece '" + piece.getPiece() + "' with id " + output);
            }
            if (type == SentencepieceModel.ModelProto.SentencePiece.Type.UNKNOWN) {
                if (tmpUnk >= 0) {
                    throw new IllegalStateException("Unknown defined twice, for ids " + tmpUnk + " and " + i);
                } else {
                    tmpUnk = i;
                }
            }
            if (type == SentencepieceModel.ModelProto.SentencePiece.Type.USER_DEFINED) {
                userDefinedSymbols.add(piece.getPiece());
            }
            if (type == SentencepieceModel.ModelProto.SentencePiece.Type.BYTE) {
                if (!proto.getTrainerSpec().getByteFallback()) {
                    throw new IllegalStateException("Bytes found but byte fallback disabled");
                } else {
                    int byteVal = pieceToByte(piece.getPiece());
                    if (byteVal < 0 || byteVal > 256) {
                        throw new IllegalStateException("Invalid byte piece found with value " + byteVal);
                    } else {
                        bytes[byteVal] = true;
                    }
                }
            }
        }
        if (proto.getTrainerSpec().getByteFallback()) {
            // validate all bytes found
            for (boolean aByte : bytes) {
                if (!aByte) {
                    throw new IllegalStateException("Byte fallback enabled but not all bytes found");
                }
            }
        }
        this.inverseVocab = new String[vocab.size() + reservedIdMap.size()];
        this.inverseVocabUTF8 = new byte[vocab.size() + reservedIdMap.size()][];
        for (var e : vocab.entrySet()) {
            int id = e.getValue();
            if ((id < 0) || (id >= inverseVocab.length)) {
                throw new IllegalStateException("Invalid vocab element, expected id in range [0, " + inverseVocab.length + "], found " + id);
            } else if (inverseVocab[id] != null) {
                throw new IllegalStateException("Invalid vocab, two elements map to id " + id + ", terms '" + inverseVocab[id] + "' & '" + e.getKey() + "'");
            } else {
                inverseVocab[id] = e.getKey();
                inverseVocabUTF8[id] = UTF8Utils.UTF8.encode(e.getKey()).array();
            }
        }
        for (var e : reservedIdMap.entrySet()) {
            int id = e.getValue();
            if ((id < 0) || (id >= inverseVocab.length)) {
                throw new IllegalStateException("Invalid vocab element, expected id in range [0, " + inverseVocab.length + "], found " + id);
            } else if (inverseVocab[id] != null) {
                throw new IllegalStateException("Invalid vocab, two elements map to id " + id + ", terms '" + inverseVocab[id] + "' & '" + e.getKey() + "'");
            } else {
                inverseVocab[id] = e.getKey();
                inverseVocabUTF8[id] = UTF8Utils.UTF8.encode(e.getKey()).array();
            }
        }
        this.proto = proto;
        this.unkId = proto.getTrainerSpec().hasUnkId() ? proto.getTrainerSpec().getUnkId() : DEFAULT_UNKNOWN_ID;
        if (tmpUnk != unkId) {
            throw new IllegalStateException("Invalid sentence piece, multiple unks found with ids " + tmpUnk + " and " + unkId);
        }
        this.bosId = proto.getTrainerSpec().hasBosId() ? proto.getTrainerSpec().getBosId() : DEFAULT_BOS_ID;
        this.eosId = proto.getTrainerSpec().hasEosId() ? proto.getTrainerSpec().getEosId() : DEFAULT_EOS_ID;
        this.padId = proto.getTrainerSpec().hasPadId() ? proto.getTrainerSpec().getPadId() : DEFAULT_PAD_ID;
        this.unk = proto.getTrainerSpec().hasUnkPiece() ? proto.getTrainerSpec().getUnkPiece() : DEFAULT_UNKNOWN;
        this.bos = proto.getTrainerSpec().hasBosPiece() ? proto.getTrainerSpec().getBosPiece() : DEFAULT_BOS;
        this.eos = proto.getTrainerSpec().hasEosPiece() ? proto.getTrainerSpec().getEosPiece() : DEFAULT_EOS;
        this.pad = proto.getTrainerSpec().hasPadPiece() ? proto.getTrainerSpec().getPadPiece() : DEFAULT_PAD;
        this.byteFallback = proto.getTrainerSpec().getByteFallback();

        this.userDefinedSymbolTrie = new Trie(userDefinedSymbols);
        this.normalizer = new Normalizer(proto.getNormalizerSpec(), proto.getTrainerSpec().getTreatWhitespaceAsSuffix(), userDefinedSymbolTrie);
        this.denormalizer = proto.hasDenormalizerSpec() ? new Normalizer(proto.getDenormalizerSpec()) : null;
        this.options = options;
    }

    public int getVocabSize() {
        return vocab.size();
    }

    public int getIdForVocab(String vocabElement) {
        var output = vocab.get(vocabElement);
        if (output == null) {
            output = reservedIdMap.get(vocabElement);
            if (output == null) {
                return 0;
            } else {
                return output;
            }
        } else {
            return output;
        }
    }

    public String getVocabForId(int id) {
        if ((id < 0) || (id >= inverseVocab.length)) {
            return unk;
        } else {
            return inverseVocab[id];
        }
    }

    byte[] getVocabBytesForId(int id) {
        if ((id < 0) || (id >= inverseVocab.length)) {
            return unk.getBytes(UTF8Utils.UTF8);
        } else {
            return inverseVocabUTF8[id];
        }
    }

    public float getScore(int id) {
        if ((id < 0) || (id >= proto.getPiecesCount())) {
            return Float.NEGATIVE_INFINITY;
        } else {
            return proto.getPieces(id).getScore();
        }
    }

    public boolean isUnknown(int id) {
        return id == unkId;
    }

    public boolean isControl(int id) {
        return proto.getPieces(id).getType().equals(SentencepieceModel.ModelProto.SentencePiece.Type.CONTROL);
    }

    public boolean isUnused(int id) {
        return proto.getPieces(id).getType().equals(SentencepieceModel.ModelProto.SentencePiece.Type.UNUSED);
    }

    public boolean isByte(int id) {
        return proto.getPieces(id).getType().equals(SentencepieceModel.ModelProto.SentencePiece.Type.BYTE);
    }

    public boolean isUserDefined(int id) {
        return proto.getPieces(id).getType().equals(SentencepieceModel.ModelProto.SentencePiece.Type.USER_DEFINED);
    }

    public int getUnknownId() {
        return unkId;
    }

    public int getBOSId() {
        return bosId;
    }

    public int getEOSId() {
        return eosId;
    }

    public int getPadId() {
        return padId;
    }

    public String getUnknown() {
        return unk;
    }

    public String getBOS() {
        return bos;
    }

    public String getEOS() {
        return eos;
    }

    public String getPad() {
        return pad;
    }

    public List<String> encode(String input) {
        return encodeToTokens(input).stream().map(SPToken::piece).toList();
    }

    public int[] encodeToInts(String input) {
        Normalizer.NormalizedOutput normalized = normalizer.normalize(input);
        return encodeToTokens(normalized).stream().mapToInt(SPToken::id).toArray();
    }

    public List<SPToken> encodeToTokens(String input) {
        Normalizer.NormalizedOutput normalized = normalizer.normalize(input);
        return encodeToTokens(normalized);
    }

    protected abstract SPPair[] encode(ByteBuffer input, boolean addBOS, boolean addEOS);

    protected List<SPToken> encodeToTokens(Normalizer.NormalizedOutput input) {
        SPPair[] pieces = encode(input.output(), options.contains(ExtraOptions.ADD_BOS), options.contains(ExtraOptions.ADD_EOS));

        List<SPToken> output = new ArrayList<>();

        int consumed = 0;
        boolean isPrevUnk = false;
        for (SPPair piece : pieces) {
            int id = piece.id();
            byte[] pieceBytes = piece.bytes();

            boolean isUnk = isUnknown(id);

            if (isControl(id)) {
                // Control symbol has no corresponding source surface, so begin == end.
                SPToken cur = new SPToken(pieceBytes, id, new byte[0], input.byteAlignment()[consumed], input.byteAlignment()[consumed]);
                output.add(cur);
            } else {
                final int begin = consumed;
                final int end = consumed + pieceBytes.length;
                if (input.byteAlignment().length < begin || input.byteAlignment().length < end) {
                    throw new IllegalStateException("Normalizer mapping invalid, begin = " + begin + ", end = " + end + ", mapping.length = " + input.byteAlignment().length);
                }
                final int origBegin = input.byteAlignment()[begin];
                final int origEnd = input.byteAlignment()[end];
                if (origEnd >= input.input().capacity() || origBegin >= input.input().capacity() || origBegin > origEnd) {
                    throw new IllegalStateException("Normalizer mapping invalid, input.length = " + input.input().capacity() + ", begin = " + begin + ", end = " + end);
                }
                ByteBuffer surface = input.input().slice(origBegin, origEnd - origBegin);

                if (isUnk && byteFallback) {
                    // Decomposes an unknown piece into UTF-8 bytes
                    for (int j = 0; j < pieceBytes.length; ++j) {
                        // Create a byte piece
                        byte b = pieceBytes[j];
                        String bytePiece = byteToPiece(b);
                        int newId = getIdForVocab(bytePiece);

                        SPToken token;
                        // The last byte piece holds the surface of the original unknown
                        // character. The other byte pieces have no surface.
                        if (j == pieceBytes.length - 1) {
                            token = new SPToken(bytePiece, newId, surface, origBegin, origEnd);
                        } else {
                            // begin == end
                            token = new SPToken(bytePiece, newId, "", origBegin, origBegin);
                        }
                        output.add(token);
                    }
                } else {
                    if (isPrevUnk && isUnk) {
                        // Overwrite the last unknown with a bigger unknown which contains both of them
                        SPToken oldToken = output.get(output.size() - 1);
                        SPToken newToken = new SPToken(pieceBytes, id, surface, origBegin, origEnd);
                        SPToken merged = SPToken.merge(oldToken, newToken);
                        output.set(output.size() - 1, merged);
                    } else {
                        SPToken token = new SPToken(pieceBytes, id, surface, origBegin, origEnd);
                        output.add(token);
                    }
                }
                consumed += pieceBytes.length;
            }
            isPrevUnk = isUnk;
        }

        if (consumed != input.output().capacity()) {
            throw new IllegalStateException("All normalized characters are not consumed");
        }

        if (options.contains(ExtraOptions.REVERSE)) {
            Collections.reverse(output);
        }

        return output;
    }

    public String decode(List<String> input) {
        int[] ints = new int[input.size()];
        for (int i = 0; i < ints.length; i++) {
            ints[i] = getIdForVocab(input.get(i));
        }
        return decodeFromInts(ints);
    }

    public String decodeFromInts(int[] input) {
        ByteBuffer output = innerDecodeFromInts(input);
        if (denormalizer != null) {
            output = denormalizer.normalize(output).output();
        }
        return UTF8Utils.UTF8.decode(output).toString();
    }

    protected ByteBuffer innerDecodeFromInts(int[] input) {

    }

    public String decodeFromTokens(List<SPToken> input) {
        int[] ints = new int[input.size()];
        for (int i = 0; i < ints.length; i++) {
            ints[i] = input.get(i).id();
        }
        return decodeFromInts(ints);
    }

    public static String byteToPiece(int byteVal) {
        if (byteVal < 256 && byteVal >= 0) {
            return HEX_FORMATTER.formatHex(new byte[]{(byte) byteVal});
        } else {
            throw new IllegalArgumentException("Invalid byte " + byteVal);
        }
    }

    public static int pieceToByte(String piece) {
        try {
            return 0xFF & (int) HEX_FORMATTER.parseHex(piece)[0];
        } catch (IllegalArgumentException e) {
            return -1;
        }
    }

    public static SPModel loadFromProto(Path protoPath, EnumSet<ExtraOptions> options) throws IOException {
        try (var fis = Files.newInputStream(protoPath)) {
            return loadFromProto(fis, options);
        }
    }

    public static SPModel loadFromProto(InputStream protoStream, EnumSet<ExtraOptions> options) throws IOException {
        return loadFromProto(SentencepieceModel.ModelProto.parseFrom(protoStream), options);
    }

    public static SPModel loadFromProto(SentencepieceModel.ModelProto proto, EnumSet<ExtraOptions> options) {
        var type = proto.getTrainerSpec().getModelType();
        return switch (type) {
            case UNIGRAM -> new UnigramSPModel(proto, options);
            case BPE -> new BPESPModel(proto, options);
            case WORD -> new WordSPModel(proto, options);
            case CHAR -> new CharSPModel(proto, options);
        };
    }

    protected record SPPair(int id, byte[] bytes) {}
}
