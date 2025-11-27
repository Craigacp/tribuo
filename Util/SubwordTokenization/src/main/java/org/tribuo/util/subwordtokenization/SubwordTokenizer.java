/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.util.subwordtokenization;

import java.util.Map;
import java.util.Set;

public interface SubwordTokenizer {

    public int getUnkTokenID();

    public int getBOSTokenID();

    public int getEOSTokenID();

    public int getPadTokenID();

    public String getUnkToken();

    public String getBOSToken();

    public String getEOSToken();

    public String getPadToken();

    public Set<String> getVocab();

    public Map<String, Integer> getVocabMapping();

    public String decode(int[] ids);

    public String decode(IntTensorBuffer buffer);

    public int[][] encode(String[] inputs, boolean insertSpecialTokens, boolean truncate);

    public IntTensorBuffer encode(String[] inputs, boolean insertSpecialTokens);

    public String[][] split(String[] inputs, boolean insertSpecialTokens, boolean truncate);

    public int[] mapTokensToIDs(String[] tokens);

    public SubwordToken[][] tokenize(String[] inputs, boolean insertSpecialTokens, boolean truncate);

    public record SubwordToken(String token, int startOffset, int endOffset, int id) {}

}
