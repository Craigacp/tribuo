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
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;

public abstract sealed class SPModel permits BPESPModel, CharSPModel, WordSPModel, UnigramSPModel {
    protected static final Charset UTF8 = StandardCharsets.UTF_8;
    protected static final HexFormat HEX_FORMATTER = HexFormat.of().withPrefix("<0x").withSuffix(">").withUpperCase();

    public enum ExtraOptions {REVERSE, ADD_BOS, ADD_EOS, UNK }

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

    protected final Map<String, Integer> reservedIdMap;

    protected final PrefixMatcher userDefinedSymbolMatcher;

    protected final int unkId;
    protected final int bosId;
    protected final int eosId;
    protected final int padId;

    protected final String unk;
    protected final String bos;
    protected final String eos;
    protected final String pad;

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
        for (var e : vocab.entrySet()) {
            int id = e.getValue();
            if ((id < 0) || (id >= inverseVocab.length)) {
                throw new IllegalStateException("Invalid vocab element, expected id in range [0, " + inverseVocab.length + "], found " + id);
            } else if (inverseVocab[id] != null) {
                throw new IllegalStateException("Invalid vocab, two elements map to id " + id + ", terms '" + inverseVocab[id] + "' & '" + e.getKey() + "'");
            } else {
                inverseVocab[id] = e.getKey();
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

        this.userDefinedSymbolMatcher = new PrefixMatcher(userDefinedSymbols);
        this.normalizer = new Normalizer(proto.getNormalizerSpec(), proto.getTrainerSpec().getTreatWhitespaceAsSuffix(), userDefinedSymbolMatcher);
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
        return encodeToTokens(input).stream().map(SPToken::surface).toList();
    }

    public int[] encodeToInts(String input) {
        Normalizer.NormalizedOutput normalized = normalizer.normalize(input);
        return encodeToInts(normalized.output(), options.contains(ExtraOptions.ADD_BOS), options.contains(ExtraOptions.ADD_EOS));
    }

    public List<SPToken> encodeToTokens(String input) {
        Normalizer.NormalizedOutput normalized = normalizer.normalize(input);
        return encodeToTokens(normalized.output(), options.contains(ExtraOptions.ADD_BOS), options.contains(ExtraOptions.ADD_EOS));
    }

    protected abstract int[] encodeToInts(String input, boolean addBOS, boolean addEOS);

    protected List<SPToken> encodeToTokens(String input, boolean addBOS, boolean addEOS) {
        int[] ints = encodeToInts(input, addBOS, addEOS);


    }

    public String decode(List<String> input) {
        int[] ints = new int[input.size()];
        for (int i = 0; i < ints.length; i++) {
            ints[i] = getIdForVocab(input.get(i));
        }
        return decodeFromInts(ints);
    }

    public String decodeFromInts(int[] input) {
        String output = innerDecodeFromInts(input);
        return denormalizer == null ? output : denormalizer.normalize(output).output();
    }

    protected abstract String innerDecodeFromInts(int[] input);

    public String decodeFromTokens(List<SPToken> input) {
        int[] ints = new int[input.size()];
        for (int i = 0; i < ints.length; i++) {
            ints[i] = input.get(i).id();
        }
        return decodeFromInts(ints);
    }

    public static String byteToPiece(int byteVal) {
        if (byteVal < 256 && byteVal >=0 ) {
            return HEX_FORMATTER.formatHex(new byte[]{(byte)byteVal});
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
}
