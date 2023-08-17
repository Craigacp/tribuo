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
        return normalize(SPModel.UTF8.encode(input));
    }

    public NormalizedOutput normalize(ByteBuffer input) {

    }

    private static SplitCharMap decodePrecompiledCharsMap(ByteBuffer buffer) {
        // <trie size(4byte)><double array trie><normalized string>
        int trieSize = buffer.getInt();

        ByteBuffer trieBuffer = ByteBuffer.allocate(trieSize);
        ByteBuffer normalizedBuffer = ByteBuffer.allocate((buffer.capacity() - trieSize) - 4);
        trieBuffer.put(0, buffer, 4, trieSize);
        normalizedBuffer.put(0, buffer, trieSize+4, normalizedBuffer.capacity());

        return new SplitCharMap(trieBuffer, normalizedBuffer);
    }

    public record NormalizedOutput(ByteBuffer output, int[] byteAlignment) {}

    private record SplitCharMap(ByteBuffer trie, ByteBuffer normalized) {}

}
