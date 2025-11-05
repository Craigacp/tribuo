/*
 * Copyright (c) 2023, 2025, Oracle and/or its affiliates. All rights reserved.
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
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;

/**
 * Whitespace separated tokenizer.
 */
public final class WordSPModel extends SPModel {

    /**
     * Construct a whitespace separated tokenizer using the supplied proto and options.
     * @param proto The tokenizer proto.
     * @param options The tokenizer options.
     */
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

        List<SPPair> tokens = new ArrayList<>();
        if (addBOS) {
            tokens.add(bosPair);
        }
        int curStart = 0;
        for (int i = 0; i < input.remaining(); i++) {
            if (input.get(i) == Normalizer.REPLACEMENT_SPACE_ARR[0] && (i+Normalizer.REPLACEMENT_SPACE_ARR.length < input.remaining())) {
                // Could be a replacement space, check the next bytes.
                if (input.get(i+1) == Normalizer.REPLACEMENT_SPACE_ARR[1] && input.get(i+2) == Normalizer.REPLACEMENT_SPACE_ARR[2]) {
                    // If a valid space character emit curStart -> i-1 as a token, set
                    // curStart to i, continue.
                    ByteBuffer slice = input.slice(curStart, i - curStart);
                    int id = getIdForVocab(UTF8Utils.UTF8.decode(slice).toString());
                    tokens.add(new SPPair(id, slice.array()));
                    curStart = i;
                }
            }
        }
        // Deal with trailing token if it exists
        if (curStart != input.remaining()) {
            ByteBuffer slice = input.slice(curStart, input.remaining()-curStart);
            int id = getIdForVocab(UTF8Utils.UTF8.decode(slice).toString());
            tokens.add(new SPPair(id, slice.array()));
        }
        if (addEOS) {
            tokens.add(eosPair);
        }

        return tokens.toArray(new SPPair[0]);
    }

}
