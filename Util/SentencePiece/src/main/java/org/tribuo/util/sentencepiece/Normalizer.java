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

import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.List;
import org.tribuo.util.sentencepiece.Trie.PrefixMatch;
import org.tribuo.util.sentencepiece.Trie.TrieResult;
import org.tribuo.util.sentencepiece.UTF8Utils.UTFCodepoint;
import org.tribuo.util.sentencepiece.protos.SentencepieceModel;

import java.nio.ByteBuffer;

/**
 * Normalizes input text according to a precompiled SentencePiece character map, applying
 * Unicode normalization rules, whitespace handling, and optional escape of whitespace to
 * the replacement space character (U+2581).
 */
public final class Normalizer {
    static final char REPLACEMENT_SPACE_CODEPOINT = '\u2581';
    static final byte[] REPLACEMENT_SPACE_ARR = new byte[]{(byte)0xE2,(byte)0x96,(byte)0x81};
    private static final byte SPACE_BYTE = ' ';
    private static final byte[] SPACE_ARR = new byte[]{SPACE_BYTE};

    private final Trie unicodeNormalizer;
    private final Trie matcher;
    private final boolean treatWhitespaceAsSuffix;

    private final ByteBuffer normalizedOutput;
    private final SentencepieceModel.NormalizerSpec proto;

    /**
     * Constructs a normalizer from the given spec, with whitespace treated as a prefix and
     * no user-defined symbol matching.
     * @param proto The normalizer spec from the SentencePiece model protobuf.
     */
    public Normalizer(SentencepieceModel.NormalizerSpec proto) {
        this(proto, false, null);
    }

    /**
     * Constructs a normalizer from the given spec.
     * @param proto The normalizer spec from the SentencePiece model protobuf.
     * @param treatWhitespaceAsSuffix If true, the dummy whitespace prefix is added as a suffix instead.
     * @param matcher A trie of user-defined symbols that are passed through unchanged during normalization,
     *                or null if there are no user-defined symbols.
     */
    public Normalizer(SentencepieceModel.NormalizerSpec proto, boolean treatWhitespaceAsSuffix, Trie matcher) {
        this.proto = proto;
        this.treatWhitespaceAsSuffix = treatWhitespaceAsSuffix;
        this.matcher = matcher;
        var charMap = decodePrecompiledCharsMap(proto.getPrecompiledCharsmap().asReadOnlyByteBuffer());
        this.normalizedOutput = charMap.normalized;
        this.unicodeNormalizer = new Trie(charMap.trie());
    }

    /**
     * Normalizes the input string and returns the normalized output along with the byte alignment mapping.
     * @param input The string to normalize.
     * @return The normalized output.
     */
    public NormalizedOutput normalize(String input) {
        return normalize(UTF8Utils.UTF8.encode(input).asReadOnlyBuffer());
    }

    /**
     * Normalizes the input UTF-8 byte buffer and returns the normalized output along with the byte
     * alignment mapping from normalized bytes back to positions in the original input buffer.
     * @param input The UTF-8 encoded input to normalize.
     * @return The normalized output.
     */
    public NormalizedOutput normalize(ByteBuffer input) {
        if (!input.hasRemaining()) {
            return new NormalizedOutput(ByteBuffer.allocate(0), ByteBuffer.allocate(0), new int[0]);
        }

        int consumed = 0;

        // Ignores leading space.
        if (proto.getRemoveExtraWhitespaces()) {
            while (input.hasRemaining()) {
                var p = normalizePrefix(input);
                if (p.output.get(0) != SPACE_BYTE) {
                    break;
                }
                input.position(input.position() + p.length);
                consumed += p.length;
            }
            // All codepoints are whitespace, return an empty buffer.
            if (!input.hasRemaining()) {
                return new NormalizedOutput(ByteBuffer.allocate(0), ByteBuffer.allocate(0), new int[0]);
            }
        }

        ByteBuffer output = ByteBuffer.allocate(input.remaining() * 3);
        IntBuffer mapping = IntBuffer.allocate(input.remaining() * 3);

        // Adds a space symbol as a prefix (default is true)
        if (!treatWhitespaceAsSuffix && proto.getAddDummyPrefix()){
            if (proto.getEscapeWhitespaces()) {
                // Replaces white space with U+2581 (LOWER ONE EIGHT BLOCK)
                output.put(REPLACEMENT_SPACE_ARR);
                for (int i = 0; i < REPLACEMENT_SPACE_ARR.length; i++) {
                    mapping.put(consumed);
                }
            } else {
                output.put((byte) ' ');
                mapping.put(consumed);
            }
        }

        boolean isPrevSpace = proto.getRemoveExtraWhitespaces();
        while (input.hasRemaining()) {
            var p = normalizePrefix(input);

            // Removes leading spaces if the previous piece ends with whitespace.
            while (isPrevSpace && p.output.hasRemaining() && p.output.get() == SPACE_BYTE) { }

            if (p.output().hasRemaining()) {
                byte cur = p.output.get(p.output.position());
                while (p.output.hasRemaining()) {
                    cur = p.output.get();
                    if (proto.getEscapeWhitespaces() && cur == ' ') {
                        // replace ' ' with kSpaceSymbol.
                        output.put(REPLACEMENT_SPACE_ARR);
                        for (int i = 0; i < REPLACEMENT_SPACE_ARR.length; i++) {
                            mapping.put(consumed);
                        }
                    } else {
                        output.put(cur);
                        mapping.put(consumed);
                    }
                }
                // Checks whether the last codepoint is whitespace.
                isPrevSpace = cur == ' ';
            }

            consumed += p.length;
            input.position(input.position()+p.length);
            if (!proto.getRemoveExtraWhitespaces()) {
                isPrevSpace = false;
            }
        }

        // Ignores trailing space.
        if (proto.getRemoveExtraWhitespaces()) {
            byte[] space = proto.getEscapeWhitespaces() ? REPLACEMENT_SPACE_ARR : SPACE_ARR;
            while (bufferEndsWith(output, space)) {
                int length = output.position() - space.length;
                if (length < 0) {
                    throw new IllegalStateException("Too many spaces after normalizing");
                }
                consumed = mapping.get(length);
                output.position(length);
                mapping.position(length);
            }
        }

        // Adds a space symbol as a suffix
        if (treatWhitespaceAsSuffix && proto.getAddDummyPrefix()) {
            if (proto.getEscapeWhitespaces()) {
                // Replaces white space with U+2581 (LOWER ONE EIGHT BLOCK)
                output.put(REPLACEMENT_SPACE_ARR);
                for (int i = 0; i < REPLACEMENT_SPACE_ARR.length; i++) {
                    mapping.put(consumed);
                }
            } else {
                output.put((byte) ' ');
                mapping.put(consumed);
            }
        }

        mapping.put(consumed);

        if (mapping.position() != (output.position() + 1)) {
            throw new IllegalStateException("Invalid normalization");
        }

        ByteBuffer slicedOutput = output.slice(0, output.position());
        int[] mappingArr = new int[mapping.position()];
        mapping.get(0, mappingArr,0, mapping.position());
        return new NormalizedOutput(input, slicedOutput, mappingArr);
    }

    /**
     * Normalizes the longest prefix of the input buffer and returns the normalized bytes along with
     * the number of input bytes consumed. If a user-defined symbol matches at the current position it
     * is returned verbatim. Otherwise the precompiled Unicode normalization trie is consulted, and if
     * no rule matches the next UTF-8 codepoint is passed through unchanged (or replaced with U+FFFD if
     * it is malformed).
     * @param input The UTF-8 input buffer, read from the current position.
     * @return A pair of the normalized output bytes and the number of input bytes consumed.
     */
    private NormalizerPair normalizePrefix(ByteBuffer input) {
        if (input.remaining() == 0) {
            return new NormalizerPair(null, 0);
        }

        if (matcher != null) {
            PrefixMatch matches = matcher.longestPrefixMatch(input);
            if (matches.found()) {
                return new NormalizerPair(input.slice(input.position(), matches.lengthConsumed()), matches.lengthConsumed());
            }
        }

        int longestLength = 0;
        int longestValue = 0;

        List<TrieResult> results = unicodeNormalizer.commonPrefixSearch(input);

        // Finds the longest rule.
        for (var r : results) {
            if (longestLength == 0 || r.length() > longestLength) {
                longestLength = r.length();  // length of prefix
                longestValue = r.value();    // index into normalized.
            }
        }

        if (longestLength == 0) {
            UTFCodepoint decode = UTF8Utils.decodeOneCodepoint(input);
            if (decode.valid()) {
                // Found a malformed utf8 codepoint, return 0xFFFD
                return new NormalizerPair(ByteBuffer.wrap(Arrays.copyOf(UTF8Utils.REPLACEMENT_CHAR_ARR, UTF8Utils.REPLACEMENT_CHAR_ARR.length)), 1);
            } else {
                return new NormalizerPair(input.slice(input.position(), decode.length()), decode.length());
            }
        } else {
            return new NormalizerPair(sliceNormalized(longestValue), longestLength);
        }
    }

    /**
     * Slices out a ByteBuffer from {@link #normalizedOutput}, which is delimited by the
     * zero byte.
     * @param index The start index.
     * @return A ByteBuffer.
     */
    private ByteBuffer sliceNormalized(int index) {
        if (index > normalizedOutput.capacity()) {
            throw new IllegalArgumentException("Invalid index, outside capacity, found " + index + ", expected < " + normalizedOutput.capacity());
        }

        for (int i = index; i < normalizedOutput.capacity(); i++) {
            // search for zero byte
            if (normalizedOutput.get(i) == 0) {
                return normalizedOutput.slice(index, i-index);
            }
        }
        return normalizedOutput.slice(index, normalizedOutput.capacity() - index);
    }

    /**
     * Returns true if the bytes written so far into {@code buffer} (up to its current position)
     * end with the given byte sequence.
     * @param buffer The buffer to check, using its current position as the logical end.
     * @param array The byte sequence to look for at the end.
     * @return True if the filled portion of the buffer ends with {@code array}.
     */
    private static boolean bufferEndsWith(ByteBuffer buffer, byte[] array) {
        if (buffer.position() >= array.length) {
            boolean match = true;
            for (int i = 0; i < array.length; i++) {
                match &= buffer.get(buffer.position()-i) == array[(array.length-1)-i];
            }
            return match;
        } else {
            return false;
        }
    }

    /**
     * Parses the precompiled character map from the SentencePiece model protobuf.
     * The layout is: {@code trie size (4 bytes little-endian),trie bytes,normalized string bytes}.
     * @param buffer The raw precompiled character map bytes.
     * @return A record holding the trie buffer and the normalized string table buffer.
     */
    private static SplitCharMap decodePrecompiledCharsMap(ByteBuffer buffer) {
        // <trie size(4byte)><trie><normalized string>
        int trieSize = buffer.getInt();

        ByteBuffer trieBuffer = ByteBuffer.allocate(trieSize);
        ByteBuffer normalizedBuffer = ByteBuffer.allocate((buffer.capacity() - trieSize) - 4);
        trieBuffer.put(0, buffer, 4, trieSize);
        normalizedBuffer.put(0, buffer, trieSize+4, normalizedBuffer.capacity());

        return new SplitCharMap(trieBuffer.asReadOnlyBuffer(), normalizedBuffer.asReadOnlyBuffer());
    }

    /**
     * The result of normalizing a string.
     * @param input The original input as a UTF-8 byte buffer.
     * @param output The normalized output as a UTF-8 byte buffer.
     * @param byteAlignment An array of length {@code output.capacity() + 1} mapping each byte position
     *                      in the normalized output back to its corresponding byte position in the original
     *                      input. The final entry maps the position one past the end of the output.
     */
    public record NormalizedOutput(ByteBuffer input, ByteBuffer output, int[] byteAlignment) {}

    /**
     * The two halves of the precompiled character map blob.
     * @param trie The serialised double-array trie used to match input prefixes.
     * @param normalized The concatenated normalized replacement strings, each terminated by a zero byte,
     *                   indexed by the values stored in the trie.
     */
    private record SplitCharMap(ByteBuffer trie, ByteBuffer normalized) {}

    /**
     * The normalized output for a single input prefix together with the number of input bytes consumed.
     * @param output The normalized UTF-8 bytes for this prefix.
     * @param length The number of bytes consumed from the input buffer.
     */
    private record NormalizerPair(ByteBuffer output, int length) {}
}
