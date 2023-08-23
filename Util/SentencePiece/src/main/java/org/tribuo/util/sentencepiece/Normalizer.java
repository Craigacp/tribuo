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

import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.List;
import org.tribuo.util.sentencepiece.Trie.PrefixMatch;
import org.tribuo.util.sentencepiece.Trie.TrieResult;
import org.tribuo.util.sentencepiece.UTF8Utils.UTFCodepoint;
import org.tribuo.util.sentencepiece.protos.SentencepieceModel;

import java.nio.ByteBuffer;

public final class Normalizer {
    static final char REPLACEMENT_SPACE_CODEPOINT = '\u2581';
    private static final byte[] REPLACEMENT_SPACE_ARR = new byte[]{(byte)0xE2,(byte)0x96,(byte)0x81};
    private static final byte SPACE_BYTE = ' ';
    private static final byte[] SPACE_ARR = new byte[]{SPACE_BYTE};

    private final Trie unicodeNormalizer;
    private final Trie matcher;
    private final boolean treatWhitespaceAsSuffix;

    private final ByteBuffer normalizedOutput;
    private final SentencepieceModel.NormalizerSpec proto;

    public Normalizer(SentencepieceModel.NormalizerSpec proto) {
        this(proto, false, null);
    }

    public Normalizer(SentencepieceModel.NormalizerSpec proto, boolean treatWhitespaceAsSuffix, Trie matcher) {
        this.proto = proto;
        this.treatWhitespaceAsSuffix = treatWhitespaceAsSuffix;
        this.matcher = matcher;
        var charMap = decodePrecompiledCharsMap(proto.getPrecompiledCharsmap().asReadOnlyByteBuffer());
        this.normalizedOutput = charMap.normalized;
        this.unicodeNormalizer = new Trie(charMap.trie());
    }

    public NormalizedOutput normalize(String input) {
        return normalize(UTF8Utils.UTF8.encode(input).asReadOnlyBuffer());
    }

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

    private static SplitCharMap decodePrecompiledCharsMap(ByteBuffer buffer) {
        // <trie size(4byte)><trie><normalized string>
        int trieSize = buffer.getInt();

        ByteBuffer trieBuffer = ByteBuffer.allocate(trieSize);
        ByteBuffer normalizedBuffer = ByteBuffer.allocate((buffer.capacity() - trieSize) - 4);
        trieBuffer.put(0, buffer, 4, trieSize);
        normalizedBuffer.put(0, buffer, trieSize+4, normalizedBuffer.capacity());

        return new SplitCharMap(trieBuffer.asReadOnlyBuffer(), normalizedBuffer.asReadOnlyBuffer());
    }

    public record NormalizedOutput(ByteBuffer input, ByteBuffer output, int[] byteAlignment) {}

    private record SplitCharMap(ByteBuffer trie, ByteBuffer normalized) {}

    private record NormalizerPair(ByteBuffer output, int length) {}
}
