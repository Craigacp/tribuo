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
            return new NormalizedOutput(ByteBuffer.allocate(0),new int[0]);
        }

        int consumed = 0;

        // Ignores heading space.
        if (proto.getRemoveExtraWhitespaces()) {
            while (input.hasRemaining()) {
                var p = normalizePrefix(input);
                if (p.output.get(0) != ' ') {
                    break;
                }
                input.position(input.position() + p.length);
                consumed += p.length;
            }
            // All chars are whitespace, return an empty buffer.
            if (!input.hasRemaining()) {
                return new NormalizedOutput(ByteBuffer.allocate(0),new int[0]);
            }
        }

        ByteBuffer output = ByteBuffer.allocate(input.remaining() * 3);
        IntBuffer mapping = IntBuffer.allocate(input.remaining() * 3);

        // Rest of normalization function.
    }

    private NormalizerPair normalizePrefix(ByteBuffer input) {
        if (input.remaining() == 0) {
            return new NormalizerPair(null, 0);
        }

        if (matcher != null) {
            PrefixMatch matches = matcher.longestPrefixMatch(input);
            if (matches.found()) {
                // TODO not yet a copy, does it need to be?
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

    private static SplitCharMap decodePrecompiledCharsMap(ByteBuffer buffer) {
        // <trie size(4byte)><double array trie><normalized string>
        int trieSize = buffer.getInt();

        ByteBuffer trieBuffer = ByteBuffer.allocate(trieSize);
        ByteBuffer normalizedBuffer = ByteBuffer.allocate((buffer.capacity() - trieSize) - 4);
        trieBuffer.put(0, buffer, 4, trieSize);
        normalizedBuffer.put(0, buffer, trieSize+4, normalizedBuffer.capacity());

        return new SplitCharMap(trieBuffer.asReadOnlyBuffer(), normalizedBuffer.asReadOnlyBuffer());
    }

    public record NormalizedOutput(ByteBuffer output, int[] byteAlignment) {}

    private record SplitCharMap(ByteBuffer trie, ByteBuffer normalized) {}

    private record NormalizerPair(ByteBuffer output, int length) {}
}
