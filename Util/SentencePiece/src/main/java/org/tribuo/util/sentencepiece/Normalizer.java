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

public final class Normalizer {

    private final Trie trie;
    private final SentencepieceModel.NormalizerSpec proto;

    public Normalizer(SentencepieceModel.NormalizerSpec proto) {
        this.proto = proto;
    }

    public NormalizedOutput normalize(String input) {

    }

    public record NormalizedOutput(String output, int[] byteAlignment) {}

}