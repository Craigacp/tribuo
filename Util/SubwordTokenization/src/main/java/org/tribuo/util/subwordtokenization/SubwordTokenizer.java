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

import org.tribuo.util.buffers.IntTensorBuffer;

import java.util.Arrays;
import java.util.Map;
import java.util.Set;

/**
 * Subword tokenization interface.
 */
public interface SubwordTokenizer {

    /**
     * The maximum length for this tokenizer.
     * @return The maximum length.
     */
    public int maxLength();

    /**
     * Int id for the Unknown token.
     * @return The unknown token id.
     */
    public int unkTokenID();

    /**
     * Int id for the beginning of sequence (BOS) token.
     * @return The BOS token id.
     */
    public int BOSTokenID();

    /**
     * Int id for the end of sequence (EOS) token.
     * @return The EOS token id.
     */
    public int EOSTokenID();

    /**
     * Int id for the padding token.
     * @return The pad token id.
     */
    public int padTokenID();

    /**
     * Surface form of the unknown token.
     * @return The unknown token.
     */
    public String unkToken();

    /**
     * Surface form of the beginning of sequence token.
     * @return The BOS token.
     */
    public String BOSToken();

    /**
     * Surface form of the end of sequence token.
     * @return The EOS token.
     */
    public String EOSToken();

    /**
     * Surface form of the padding token.
     * @return The pad token.
     */
    public String padToken();

    /**
     * The vocabulary of this tokenizer, including the special tokens.
     * @return An immutable view on the tokenizer vocabulary.
     */
    public Set<String> vocab();

    /**
     * The mapping from vocab tokens to integer ids.
     * @return The vocab to id mapping.
     */
    public Map<String, Integer> vocabMapping();

    /**
     * The mapping from integer ids to vocab tokens.
     * @return The id to vocab mapping.
     */
    public String[] reverseVocabMapping();

    /**
     * Decodes an array of token ids into the string representation, joining using the tokenizer's default
     * joiner, typically a space.
     * @param ids The array of token ids.
     * @return The string representation.
     */
    public String decode(int[] ids);

    /**
     * Decodes a buffer of token ids into the string representation, joining using the tokenizer's default
     * joiner, typically a space.
     * <p>
     * Expects the buffer to be single dimensional, will throw {@link IllegalArgumentException} otherwise.
     * @param buffer The buffer of token ids.
     * @return The string representation.
     */
    public String decode(IntTensorBuffer buffer);

    /**
     * Decodes an array of token ids to an array of String tokens.
     * @param ids The array of token ids.
     * @return The token string array.
     */
    default public String[] decodeToArray(int[] ids) {
        var mapping = reverseVocabMapping();
        String[] tokens = new String[ids.length];
        for (int i = 0; i < tokens.length; i++) {
            tokens[i] = mapping[ids[i]];
        }
        return tokens;
    }

    /**
     * Decodes a buffer of token ids to an array of String tokens.
     * <p>
     * Expects the buffer to be single dimensional, will throw {@link IllegalArgumentException} otherwise.
     * @param buffer The buffer of token ids.
     * @return The token string array.
     */
    default public String[] decodeToArray(IntTensorBuffer buffer) {
        if (buffer.shape().length != 1) {
            throw new IllegalArgumentException("The buffer must be single dimensional");
        } else if (buffer.shape()[0] > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("The buffer must have fewer than " + Integer.MAX_VALUE + " elements");
        }
        var mapping = reverseVocabMapping();
        String[] tokens = new String[(int) buffer.shape()[0]];
        for (long i = 0; i < tokens.length; i++) {
            tokens[(int)i] = mapping[buffer.get(i)];
        }
        return tokens;
    }

    /**
     * Maps an array of tokens into an array of ids.
     * @param tokens The token array, any out of vocabulary tokens will be replaced by the UNK token.
     * @return The token ids.
     */
    default public int[] mapTokensToIDs(String[] tokens) {
        var mapping = vocabMapping();
        int[] ids = new int[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            ids[i] = mapping.getOrDefault(tokens[i], unkTokenID());
        }
        return ids;
    }

    /**
     * Encodes a series of Strings into their ids, inserting special tokens and truncating if requested.
     * @param inputs The strings.
     * @param insertSpecialTokens Should special tokens like BOS and EOS be inserted?
     * @param truncate Should the token arrays be truncated to the tokenizer's max length?
     * @param padding Should the token arrays be padded to the same length?
     * @return An array of token ids.
     */
    default public int[][] encodeToArray(String[] inputs, boolean insertSpecialTokens, boolean truncate, boolean padding) {
        var tokens = tokenize(inputs, insertSpecialTokens, truncate);
        int maxLength = -1;
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i].length > maxLength) {
                maxLength = tokens[i].length;
            }
        }
        int[][] ids = new int[tokens.length][];
        for (int i = 0; i < tokens.length; i++) {
            if (padding) {
                ids[i] = new int[maxLength];
                Arrays.fill(ids[i], padTokenID());
            } else {
                ids[i] = new int[tokens[i].length];
            }
            for (int j = 0; j < tokens[i].length; j++) {
                ids[i][j] = tokens[i][j].id();
            }
        }
        return ids;
    }

    /**
     * Encodes a series of Strings into their ids, inserting special tokens and truncating if requested.
     * <p>
     * The output is always padded to the length of the longest input.
     * @param inputs The strings.
     * @param insertSpecialTokens Should special tokens like BOS and EOS be inserted?
     * @param truncate Should the token arrays be truncated to the tokenizer's max length?
     * @return An array of token ids.
     */
    default public IntTensorBuffer encodeToBuffer(String[] inputs, boolean insertSpecialTokens, boolean truncate) {
        var tokens = tokenize(inputs, insertSpecialTokens, truncate);
        int maxLength = -1;
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i].length > maxLength) {
                maxLength = tokens[i].length;
            }
        }
        IntTensorBuffer buffer = new IntTensorBuffer(new long[]{tokens.length, maxLength}, padTokenID());
        var buf = buffer.getBuffer();
        int idx;
        for (int i = 0; i < tokens.length; i++) {
            idx = i*maxLength;
            for (int j = 0; j < tokens[i].length; j++) {
                buf.put(idx, tokens[i][j].id());
                idx++;
            }
        }
        return buffer;
    }

    /**
     * Splits a series of strings into their tokens, without encoding them into ids.
     * @param inputs The strings.
     * @param insertSpecialTokens Should special tokens be inserted?
     * @param truncate Should the token arrays be truncated to the tokenizer's max length?
     * @return An array of tokens.
     */
    default public String[][] split(String[] inputs, boolean insertSpecialTokens, boolean truncate) {
        var tokens = tokenize(inputs, insertSpecialTokens, truncate);
        String[][] output = new String[tokens.length][];
        for (int i = 0; i < tokens.length; i++) {
            output[i] = new String[tokens[i].length];
            for (int j = 0; j < tokens[i].length; j++) {
                output[i][j] = tokens[i][j].token();
            }
        }
        return output;
    }

    /**
     * Encodes a String into its subword tokens including offsets, inserting special tokens and
     * truncating if requested.
     * @param input The string.
     * @param insertSpecialTokens Should special tokens like BOS and EOS be inserted?
     * @param truncate Should the token arrays be truncated to the tokenizer's max length?
     * @return An array of subword tokens.
     */
    default public SubwordToken[] tokenize(String input, boolean insertSpecialTokens, boolean truncate) {
        return tokenize(new String[]{input}, insertSpecialTokens, truncate)[0];
    }

    /**
     * Encodes a series of Strings into their subword tokens including offsets, inserting special tokens and
     * truncating if requested.
     * @param inputs The strings.
     * @param insertSpecialTokens Should special tokens like BOS and EOS be inserted?
     * @param truncate Should the token arrays be truncated to the tokenizer's max length?
     * @return An array of subword tokens.
     */
    public SubwordToken[][] tokenize(String[] inputs, boolean insertSpecialTokens, boolean truncate);

    /**
     * A subword token.
     * @param token The vocab token.
     * @param startOffset The start character offset.
     * @param endOffset The end character offset.
     * @param id The token id.
     * @param isControl Is this a control token.
     */
    public record SubwordToken(String token, int startOffset, int endOffset, int id, boolean isControl) {}

}
