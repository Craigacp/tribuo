/*
 * Copyright (c) 2015, 2024, Oracle and/or its affiliates. All rights reserved.
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
package org.tribuo.util.tokens.impl.wordpiece;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

import org.tribuo.util.tokens.Token;
import org.tribuo.util.tokens.Token.TokenType;
import org.tribuo.util.tokens.Tokenizer;
import org.tribuo.util.tokens.impl.WhitespaceTokenizer;

import com.oracle.labs.mlrg.olcut.config.Config;
import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;

/**
 * This Tokenizer is meant to be a reasonable approximation of the BertTokenizer
 * defined <a href=
 * "https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/tokenization_bert.py#L117">here</a>.
 * Please see class definition for <code>BertTokenizer</code> (the line numbers
 * may change.) Please see notes in WordpieceTokenizerTest for information about
 * how we tested the similarity between this tokenizer and the referenced python
 * implementation and for regression test examples that fail. In short, there
 * are outstanding discrepancies for texts that include Arabic and other
 * non-latin scripts that generate so many "[UNK]" tokens for an English-based
 * BPE vocabulary as to render the discrepancies as practically meaningless.
 * <p>
 * As in the reference implementation, the input text is whitespace tokenized
 * and each token is further tokenized to account for things like punctuation
 * and Chinese characters. The resulting tokens are then applied to the
 * wordpiece algorithm implemented in {@link Wordpiece} which is driven by an
 * input vocabulary which matches tokens and token suffixes as it can. Any
 * tokens that are not found in the input vocabulary are marked as "unknown".
 */
public final class WordpieceTokenizer implements Tokenizer {

    private static final Pattern accentsPattern = Pattern.compile("\\p{Mn}");

    @Config(mandatory=true, description="an instance of Wordpiece which applies the 'wordpiece' algorithm")
    private Wordpiece wordpiece;
    @Config(description="determines whether or not to lowercase the input text")
    private boolean toLowerCase = true;
    @Config(description="performs whitespace tokenization before 'basic' tokenizer is applied (see basicTokenizer)")
    private Tokenizer whitespaceTokenizer = new WhitespaceTokenizer();
    @Config(description="performs some tokenization work on the input text before the wordpiece algorithm is applied to each resulting token.")
    private Tokenizer basicTokenizer = new WordpieceBasicTokenizer();
    @Config(description="determines whether or not to strip accents/diacritics from the input text")
    private boolean stripAccents = true;
    @Config(description="a set of 'token' strings that should never be split regardless of whether they have e.g., punctuation in the middle.  No entries should have whitespace in them.")
    private Set<String> neverSplitTokens = Collections.emptySet();

    // internal state member variables
    private boolean reset;

    private Token currentToken;

    private List<Token> currentWordpieceTokens = new ArrayList<>();

    private int currentWordpieceIndex;

    /**
     * For OLCUT.
     */
    @SuppressWarnings("unused")
    private WordpieceTokenizer() {
    }

    /**
     * Constructs a wordpiece tokenizer.
     * @param wordpiece        an instance of {@link Wordpiece}
     * @param tokenizer        Wordpiece is run on the tokens generated by the
     *                         tokenizer provided here.
     * @param toLowerCase      determines whether or not to lowercase each token
     *                         before running Wordpiece on it
     * @param stripAccents     determines whether or not to strip out accents from
     *                         each token before running Wordpiece on it
     * @param neverSplit       a set of token values that we will not apply
     *                         Wordpiece to. 
     */
    public WordpieceTokenizer(Wordpiece wordpiece, Tokenizer tokenizer, boolean toLowerCase, boolean stripAccents,
            Set<String> neverSplit) {
        this.wordpiece = wordpiece;
        this.basicTokenizer = tokenizer;
        this.toLowerCase = toLowerCase;
        this.stripAccents = stripAccents;
        this.neverSplitTokens = neverSplit;
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this, "Tokenizer");
    }

    @Override
    public void reset(CharSequence cs) {
        this.reset = true;
        this.whitespaceTokenizer.reset(cs);
        this.currentWordpieceTokens.clear();
        currentWordpieceIndex = -1;
        if (this.whitespaceTokenizer.advance()) {
            this.currentToken = this.whitespaceTokenizer.getToken();
            getWordpieceTokens();
        }
    }

    @Override
    public boolean advance() {
        if (!reset) {
            throw new IllegalStateException("WordpieceTokenizer has not been reset.");
        }
        currentWordpieceIndex++;
        if (currentWordpieceIndex < currentWordpieceTokens.size()) {
            return true;
        } else if (whitespaceTokenizer.advance()) {
            currentToken = this.whitespaceTokenizer.getToken();
            getWordpieceTokens();
            currentWordpieceIndex = 0;
            if (currentWordpieceTokens.isEmpty()) {
                return advance();
            }
            return true;
        } else {
            return false;
        }
    }

    /**
     * Normalizes the text by converting it into the canonical unicode decomposition
     * and then replacing accents.
     * @param text The input text to normalize.
     * @return A normalized form of the text.
     */
    private static String normalize(String text) {
        text = Normalizer.normalize(text, Normalizer.Form.NFD);
        text = accentsPattern.matcher(text).replaceAll("");
        return text;
    }

    /**
     * Generates the wordpiece tokens from the next token.
     */
    private void getWordpieceTokens() {
        this.currentWordpieceTokens.clear();

        String text = currentToken.text;
        if(neverSplitTokens.contains(text)) {
            currentWordpieceTokens.add(currentToken);
            return;
        }
        
        List<Token> basicTokens = this.basicTokenizer.tokenize(text);
        for(Token basicToken : basicTokens) {
            
            text = basicToken.text;
            
            if (toLowerCase) {
                text = text.toLowerCase();
            }
    
            if (this.stripAccents) {
                text = normalize(text);
            }
    
            List<String> wordpieces = wordpiece.wordpiece(text);
    
            if (wordpieces.isEmpty()) {
                return;
            } else if (wordpieces.size() == 1) {
                String wp = wordpieces.get(0);
                int start = basicToken.start + currentToken.start;
                int end = basicToken.end + currentToken.start;
                if (wp.equals(this.wordpiece.getUnknownToken())) {
                    currentWordpieceTokens.add(new Token(wp, start, end, TokenType.UNKNOWN));
                } else {
                    currentWordpieceTokens.add(new Token(wp, start, end, TokenType.WORD));
                }
            } else {
                int begin = currentToken.start + basicToken.start;
                for (String wp : wordpieces) {
                    TokenType type = TokenType.PREFIX;
                    int end = begin + wp.length();
                    if (wp.startsWith("##")) {
                        end -= 2;
                        type = TokenType.SUFFIX;
                    }
                    currentWordpieceTokens.add(new Token(wp, begin, end, type));
                    begin = end;
                }
            }
        }
    }

    @Override
    public Token getToken() {
        if (currentWordpieceIndex < currentWordpieceTokens.size()) {
            return currentWordpieceTokens.get(currentWordpieceIndex);
        } else {
            throw new IllegalStateException("WordpieceTokenizer is not ready.");
        }
    }

    @Override
    public String getText() {
        return getToken().text;
    }

    @Override
    public int getStart() {
        return getToken().start;
    }

    @Override
    public int getEnd() {
        return getToken().end;
    }

    @Override
    public TokenType getType() {
        return getToken().type;
    }

    @Override
    public WordpieceTokenizer clone() {
      try {
          WordpieceTokenizer copy = (WordpieceTokenizer) super.clone();
          copy.whitespaceTokenizer = whitespaceTokenizer.clone();
          copy.basicTokenizer = basicTokenizer.clone();
          copy.reset = false;
          copy.currentToken = null;
          copy.currentWordpieceTokens = new ArrayList<>();
          copy.currentWordpieceIndex = -1;
          return copy;
      } catch (CloneNotSupportedException e) {
          throw new AssertionError("WordpieceTokenizer is Cloneable, but clone call failed");
      }
    }
}