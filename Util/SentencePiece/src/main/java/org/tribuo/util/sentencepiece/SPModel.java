/*
 * Copyright (c) 2023, 2026, Oracle and/or its affiliates. All rights reserved.
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

/**
 * Base class for Sentencepiece tokenizers.
 */
public abstract sealed class SPModel permits BPESPModel, CharSPModel, UnigramSPModel, WordSPModel {

    protected static final HexFormat HEX_FORMATTER = HexFormat.of().withPrefix("<0x").withSuffix(">").withUpperCase();

    /**
     * Tokenization output options.
     */
    public enum ExtraOptions {
        /**
         * Reverse the output tokens.
         */
        REVERSE,
        /**
         * Add the BOS token to the start of the output
         */
        ADD_BOS,
        /**
         * Add the EOS token to the end of the output.
         */
        ADD_EOS,
        /**
         * Emit an unknown token if the input contains an unknown token.
         */
        UNK;
    }

    /**
     * Default unknown token.
     */
    public static final String DEFAULT_UNKNOWN = "<unk>";
    /**
     * Default BOS token.
     */
    public static final String DEFAULT_BOS = "<s>";
    /**
     * Default EOS token.
     */
    public static final String DEFAULT_EOS = "</s>";
    /**
     * Default padding token.
     */
    public static final String DEFAULT_PAD = "<pad>";
    /**
     * Default unknown token id.
     */
    public static final int DEFAULT_UNKNOWN_ID = 0;
    /**
     * Default BOS token id.
     */
    public static final int DEFAULT_BOS_ID = 1;
    /**
     * Default EOS token id.
     */
    public static final int DEFAULT_EOS_ID = 2;
    /**
     * Default pad token id.
     */
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
    protected final SPPair unkPair;
    protected final String bos;
    protected final SPPair bosPair;
    protected final String eos;
    protected final SPPair eosPair;
    protected final String pad;
    protected final SPPair padPair;
    protected final boolean byteFallback;

    protected final EnumSet<ExtraOptions> options;

    protected final Normalizer normalizer;
    protected final Normalizer denormalizer;

    protected final SentencepieceModel.ModelProto proto;

    /**
     * Builds a SPModel from the protobuf.
     * @param proto The protobuf.
     * @param options Additional options to control the output.
     */
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
        this.unkPair = new SPPair(unkId, UTF8Utils.UTF8.encode(unk).array());
        this.bosPair = new SPPair(bosId, UTF8Utils.UTF8.encode(bos).array());
        this.eosPair = new SPPair(eosId, UTF8Utils.UTF8.encode(eos).array());
        this.padPair = new SPPair(padId, UTF8Utils.UTF8.encode(pad).array());
        this.byteFallback = proto.getTrainerSpec().getByteFallback();

        this.userDefinedSymbolTrie = new Trie(userDefinedSymbols);
        this.normalizer = new Normalizer(proto.getNormalizerSpec(), proto.getTrainerSpec().getTreatWhitespaceAsSuffix(), userDefinedSymbolTrie);
        this.denormalizer = proto.hasDenormalizerSpec() ? new Normalizer(proto.getDenormalizerSpec()) : null;
        this.options = options;
    }

    /**
     * The vocab size of this tokenizer.
     * @return The vocab size.
     */
    public int getVocabSize() {
        return vocab.size();
    }

    /**
     * Returns the id number associated with this token.
     * @param vocabElement The token.
     * @return The id number, or the unknown token id if unknown.
     */
    public int getIdForVocab(String vocabElement) {
        var output = vocab.get(vocabElement);
        if (output == null) {
            return reservedIdMap.getOrDefault(vocabElement, unkId);
        } else {
            return output;
        }
    }

    /**
     * Returns the token for this id. If the id is out of bounds returns the UNK token.
     * @param id The token id.
     * @return The token.
     */
    public String getVocabForId(int id) {
        if ((id < 0) || (id >= inverseVocab.length)) {
            return unk;
        } else {
            return inverseVocab[id];
        }
    }

    /**
     * Returns the UTF-8 byte array for this token id, or the byte array for the unknown token if the id is out of bounds.
     * @param id The token id.
     * @return The token UTF-8 byte array.
     */
    byte[] getVocabBytesForId(int id) {
        if ((id < 0) || (id >= inverseVocab.length)) {
            return unk.getBytes(UTF8Utils.UTF8);
        } else {
            return inverseVocabUTF8[id];
        }
    }

    /**
     * Returns the score (log probability) of the token with the given id.
     * Returns {@link Float#NEGATIVE_INFINITY} if the id is out of range.
     * @param id The token id.
     * @return The score.
     */
    public float getScore(int id) {
        if ((id < 0) || (id >= proto.getPiecesCount())) {
            return Float.NEGATIVE_INFINITY;
        } else {
            return proto.getPieces(id).getScore();
        }
    }

    /**
     * Returns true if the given id is the unknown token id.
     * @param id The token id.
     * @return True if it is the unknown token.
     */
    public boolean isUnknown(int id) {
        return id == unkId;
    }

    /**
     * Returns true if the given id is a control token (e.g. BOS or EOS).
     * @param id The token id.
     * @return True if it is a control token.
     */
    public boolean isControl(int id) {
        return proto.getPieces(id).getType().equals(SentencepieceModel.ModelProto.SentencePiece.Type.CONTROL);
    }

    /**
     * Returns true if the given id is an unused token.
     * @param id The token id.
     * @return True if it is an unused token.
     */
    public boolean isUnused(int id) {
        return proto.getPieces(id).getType().equals(SentencepieceModel.ModelProto.SentencePiece.Type.UNUSED);
    }

    /**
     * Returns true if the given id is a byte fallback token representing a single raw byte.
     * @param id The token id.
     * @return True if it is a byte token.
     */
    public boolean isByte(int id) {
        return proto.getPieces(id).getType().equals(SentencepieceModel.ModelProto.SentencePiece.Type.BYTE);
    }

    /**
     * Returns true if the given id is a user-defined token.
     * @param id The token id.
     * @return True if it is a user-defined token.
     */
    public boolean isUserDefined(int id) {
        return proto.getPieces(id).getType().equals(SentencepieceModel.ModelProto.SentencePiece.Type.USER_DEFINED);
    }

    /**
     * Returns the unknown token id.
     * @return The unknown token id.
     */
    public int getUnknownId() {
        return unkId;
    }

    /**
     * Returns the beginning-of-sequence token id.
     * @return The BOS token id.
     */
    public int getBOSId() {
        return bosId;
    }

    /**
     * Returns the end-of-sequence token id.
     * @return The EOS token id.
     */
    public int getEOSId() {
        return eosId;
    }

    /**
     * Returns the padding token id.
     * @return The pad token id.
     */
    public int getPadId() {
        return padId;
    }

    /**
     * Returns the unknown token string.
     * @return The unknown token.
     */
    public String getUnknown() {
        return unk;
    }

    /**
     * Returns the beginning-of-sequence token string.
     * @return The BOS token.
     */
    public String getBOS() {
        return bos;
    }

    /**
     * Returns the end-of-sequence token string.
     * @return The EOS token.
     */
    public String getEOS() {
        return eos;
    }

    /**
     * Returns the padding token string.
     * @return The pad token.
     */
    public String getPad() {
        return pad;
    }

    /**
     * Tokenizes the input string and returns the token pieces as strings.
     * @param input The string to tokenize.
     * @return The list of token piece strings.
     */
    public List<String> encode(String input) {
        return encodeToTokens(input).stream().map(SPToken::piece).toList();
    }

    /**
     * Tokenizes the input string and returns the token ids.
     * @param input The string to tokenize.
     * @return The array of token ids.
     */
    public int[] encodeToInts(String input) {
        Normalizer.NormalizedOutput normalized = normalizer.normalize(input);
        return encodeToTokens(normalized).stream().mapToInt(SPToken::id).toArray();
    }

    /**
     * Tokenizes the input string and returns the full token objects including piece, id, and source span.
     * @param input The string to tokenize.
     * @return The list of tokens.
     */
    public List<SPToken> encodeToTokens(String input) {
        Normalizer.NormalizedOutput normalized = normalizer.normalize(input);
        return encodeToTokens(normalized);
    }

    /**
     * Tokenize the input UTF-8 buffer.
     * @param input UTF-8 buffer to encode. Does not respect the buffer's starting position and may change it.
     * @param addBOS Add the BOS token.
     * @param addEOS Add the EOS token.
     * @return The sentencepiece pairs representing the tokenization.
     */
    protected abstract SPPair[] encode(ByteBuffer input, boolean addBOS, boolean addEOS);

    /**
     * Tokenizes the supplied normalized input text.
     * @param input The input text.
     * @return The list of tokens.
     */
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
                    SPToken newToken = new SPToken(pieceBytes, id, surface, origBegin, origEnd);
                    if (isPrevUnk && isUnk) {
                        // Overwrite the last unknown with a bigger unknown which contains both of them
                        SPToken oldToken = output.getLast();
                        SPToken merged = SPToken.merge(oldToken, newToken);
                        output.set(output.size() - 1, merged);
                    } else {
                        output.add(newToken);
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

    /**
     * Decodes a list of token piece strings back into the original text.
     * Each piece is looked up in the vocabulary to obtain its id, then decoded via {@link #decodeFromInts(int[])}.
     * @param input The list of token piece strings to decode.
     * @return The decoded string.
     */
    public String decode(List<String> input) {
        int[] ints = new int[input.size()];
        for (int i = 0; i < ints.length; i++) {
            ints[i] = getIdForVocab(input.get(i));
        }
        return decodeFromInts(ints);
    }

    /**
     * Decodes an array of token ids back into the original text.
     * Applies the denormalizer after decoding if one is present in the model.
     * @param input The array of token ids to decode.
     * @return The decoded string.
     * @throws IllegalArgumentException If any token id is negative or out of range.
     */
    public String decodeFromInts(int[] input) {
        ByteBuffer output = innerDecodeFromInts(input);
        if (denormalizer != null) {
            output = denormalizer.normalize(output).output();
        }
        return UTF8Utils.UTF8.decode(output).toString();
    }

    /**
     * Decodes the supplied token ids into a UTF-8 encoded byte buffer.
     * <p>
     * Control tokens (BOS, EOS) are skipped. When byte fallback is enabled, consecutive
     * byte tokens are accumulated and emitted as their raw byte values rather than as
     * their piece string bytes. All other tokens emit the UTF-8 bytes of their piece
     * string. The returned buffer may still contain the replacement space character
     * (U+2581) which is converted to a regular space by the denormalizer in
     * {@link #decodeFromInts(int[])}.
     * @param input The token ids to decode.
     * @return A ByteBuffer positioned at zero containing the UTF-8 bytes of the decoded text.
     * @throws IllegalArgumentException If any token id is negative or out of range.
     */
    protected ByteBuffer innerDecodeFromInts(int[] input) {
        // Validate all ids and compute an upper bound on output size.
        // Upper bound as byte fallback tokens and control tokens contribute fewer bytes than their pieces
        int maxSize = 0;
        for (int id : input) {
            if ((id < 0 || id >= inverseVocab.length) && (id != padId)) {
                throw new IllegalArgumentException("Invalid token id: " + id + ", expected id in range [0, " + inverseVocab.length + ")");
            }
            maxSize += inverseVocabUTF8[id].length;
        }
        ByteBuffer output = ByteBuffer.allocate(maxSize);

        // Buffer for accumulating raw bytes from consecutive byte-fallback tokens
        int pendingByteCount = 0;
        byte[] pendingBytes = new byte[input.length];

        for (int id : input) {
            if (id != padId) {
                if (byteFallback && isByte(id)) {
                    // Accumulate the raw byte-fallback bytes
                    pendingBytes[pendingByteCount++] = (byte) pieceToByte(inverseVocab[id]);
                } else {
                    // Flush any pending byte-fallback bytes
                    if (pendingByteCount > 0) {
                        output.put(pendingBytes, 0, pendingByteCount);
                        pendingByteCount = 0;
                    }
                    // Emit the UTF-8 bytes of this piece (skipping control tokens)
                    if (!isControl(id)) {
                        output.put(inverseVocabUTF8[id]);
                    }
                }
            }
        }

        // Flush any remaining byte-fallback bytes
        if (pendingByteCount > 0) {
            output.put(pendingBytes, 0, pendingByteCount);
        }

        return output.slice(0, output.position());
    }

    /**
     * Decodes a list of {@link SPToken} objects back into the original text.
     * @param input The list of tokens to decode.
     * @return The decoded string.
     * @throws IllegalArgumentException If any token id is negative or out of range.
     */
    public String decodeFromTokens(List<SPToken> input) {
        int[] ints = input.stream().mapToInt(SPToken::id).toArray();
        return decodeFromInts(ints);
    }

    /**
     * Formats the supplied int as a hexadecimal number. Only accepts numbers between 0 and 255, otherwise throws {@link IllegalArgumentException}.
     * @param byteVal The input int to format.
     * @return The hexadecimal representation.
     */
    public static String byteToPiece(int byteVal) {
        if (byteVal < 256 && byteVal >= 0) {
            return HEX_FORMATTER.formatHex(new byte[]{(byte) byteVal});
        } else {
            throw new IllegalArgumentException("Invalid byte " + byteVal);
        }
    }

    /**
     * Converts the supplied string into an int by parsing it as a hexadecimal number.
     * @param piece The hexadecimal number.
     * @return An int.
     */
    public static int pieceToByte(String piece) {
        try {
            return 0xFF & (int) HEX_FORMATTER.parseHex(piece)[0];
        } catch (IllegalArgumentException e) {
            return -1;
        }
    }

    /**
     * Reads a sentencepiece model protobuf from the supplied path and constructs an {@link SPModel}.
     * @param protoPath The input path.
     * @param options Additional options to control the tokenizer output.
     * @return A SPModel instance.
     * @throws IOException If the input path could not be read, or the protobuf failed to parse.
     */
    public static SPModel loadFromProto(Path protoPath, EnumSet<ExtraOptions> options) throws IOException {
        try (var fis = Files.newInputStream(protoPath)) {
            return loadFromProto(fis, options);
        }
    }

    /**
     * Reads a sentencepiece model protobuf from the supplied input stream and constructs an {@link SPModel}.
     * @param protoStream The input stream.
     * @param options Additional options to control the tokenizer output.
     * @return A SPModel instance.
     * @throws IOException If the input stream could not be read, or the protobuf failed to parse.
     */
    public static SPModel loadFromProto(InputStream protoStream, EnumSet<ExtraOptions> options) throws IOException {
        return loadFromProto(SentencepieceModel.ModelProto.parseFrom(protoStream), options);
    }

    /**
     * Builds an SPModel subclass from the supplied sentencepiece model protobuf.
     * @param proto The input protobuf.
     * @param options Additional options to control the tokenizer output.
     * @return A SPModel instance.
     */
    public static SPModel loadFromProto(SentencepieceModel.ModelProto proto, EnumSet<ExtraOptions> options) {
        var type = proto.getTrainerSpec().getModelType();
        return switch (type) {
            case UNIGRAM -> new UnigramSPModel(proto, options);
            case BPE -> new BPESPModel(proto, options);
            case WORD -> new WordSPModel(proto, options);
            case CHAR -> new CharSPModel(proto, options);
        };
    }

    /**
     * Tuple containing the token id and the token bytes.
     * @param id The token id.
     * @param bytes The UTF-8 bytes representing the token.
     */
    protected record SPPair(int id, byte[] bytes) {}
}
