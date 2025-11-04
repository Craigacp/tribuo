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
import java.util.Arrays;
import java.util.EnumSet;
import java.util.regex.Pattern;

/**
 * Whitespace separated tokenizer.
 */
public final class WordSPModel extends SPModel {

    private static final Pattern SPLITTER = Pattern.compile("" + Normalizer.REPLACEMENT_SPACE_CODEPOINT);

    WordSPModel(SentencepieceModel.ModelProto proto, EnumSet<ExtraOptions> options) {
        super(proto, options);
    }

    @Override
    protected SPPair[] encode(ByteBuffer input, boolean addBOS, boolean addEOS) {
        if (!input.hasRemaining()) {
            if (addBOS && addEOS) {
                return new SPPair[] {bosPair, eosPair};
            } else if (addBOS) {
                return new SPPair[] {bosPair};
            } else if (addEOS) {
                return new SPPair[] {eosPair};
            } else {
                return new SPPair[0];
            }
        }

        String[] split = SPLITTER.split(input);
        int length = split.length;
        if (addBOS) {
            length++;
        }
        if (addEOS) {
            length++;
        }
        SPPair[] tokens = new SPPair[length];
        int count = 0;
        if (addBOS) {
            tokens[count] = bosPair;
            count++;
        }
        for (String s : split) {
            if (!s.isEmpty()) {
                tokens[count] = getIdForVocab(s);
                count++;
            }
        }
        if (addEOS) {
            tokens[count] = eosPair;
            count++;
        }

        return Arrays.copyOf(tokens, count);
    }

}
