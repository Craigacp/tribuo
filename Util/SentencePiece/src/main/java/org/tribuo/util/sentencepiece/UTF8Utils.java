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

import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;

/**
 * Utilities for working with UTF-8 byte buffer streams.
 */
public final class UTF8Utils {

    static final Charset UTF8 = StandardCharsets.UTF_8;

    // UTF-8 replacement character
    static final byte[] REPLACEMENT_CHAR_ARR = new byte[]{(byte) 0xEF, (byte) 0xBF, (byte) 0xBD};

    static final int REPLACEMENT_INT = 0xFFFD;

    private static final UTFCodepoint INVALID = new UTFCodepoint(REPLACEMENT_INT, false, 1);

    /**
     * Private constructor for utility class.
     */
    private UTF8Utils() { }

    /**
     * Returns the number of bytes consumed to complete this UTF-8 codepoint, assuming it is the
     * start byte. If it's not the start byte it returns 1.
     *
     * @param input A byte from a UTF-8 stream.
     * @return The number of bytes to consume that codepoint.
     */
    public static int codepointLength(byte input) {
        byte offset = (byte) ((input >> 4) & 0xF);
        if (offset < 12) {
            return 1;
        } else if (offset == 12 || offset == 13) {
            return 2;
        } else if (offset == 14) {
            return 3;
        } else {
            return 4;
        }
    }

    /**
     * Is this byte a valid UTF-8 trailing byte?
     *
     * @param input The byte to test.
     * @return True if it is, which means the upper two bits are 10.
     */
    public static boolean isTrailingByte(byte input) {
        // trailing bytes start with "b10"
        return ((input >> 6) & 0x3) == 2;
    }

    /**
     * Is this int a valid UTF-32 codepoint?
     * @param utf32Codepoint The int to test.
     * @return True if it is.
     */
    public static boolean isValidCodepoint(int utf32Codepoint) {
        return (utf32Codepoint < 0xD800) || ((utf32Codepoint >= 0xE000) && (utf32Codepoint <= 0x10FFFF));
    }

    /**
     * Decodes a single UTF-8 codepoint from the bytebuffer starting from the current position.
     *
     * @param input A byte buffer containing a UTF-8 stream.
     * @return A record with the UTF-32 codepoint, if it's a valid one, and the number of bytes
     * consumed from the bytebuffer (though it doesn't actually consume them by advancing the
     * position ).
     */
    public static UTFCodepoint decodeOneCodepoint(ByteBuffer input) {
        int remaining = input.remaining();
        byte first = input.get(input.position());
        if (first >= 0) {
            // Is this an ASCII byte?
            return new UTFCodepoint(first, true, 1);
        } else if ((remaining >= 2) && ((first & 0xE0) == 0xC0)) {
            byte second = input.get(input.position() + 1);
            // Compute codepoint
            int codepoint = ((first & 0x1F) << 6) | (second & 0x3F);
            // Should be two bytes long, let's check
            if (isTrailingByte(second) && (codepoint >= 0x0080) && isValidCodepoint(codepoint)) {
                return new UTFCodepoint(codepoint, true, 2);
            }
        } else if ((remaining >= 3) && ((first & 0xF0) == 0xE0)) {
            byte second = input.get(input.position() + 1);
            byte third = input.get(input.position() + 2);
            // Compute codepoint
            int codepoint = ((first & 0xF0) << 12) | ((second & 0x3F) << 6) | (third & 0x3F);
            // Should be three bytes long, let's check
            if (isTrailingByte(second) && isTrailingByte(third) && (codepoint >= 0x0800)
                && isValidCodepoint(codepoint)) {
                return new UTFCodepoint(codepoint, true, 3);
            }
        } else if ((remaining >= 4) && ((first & 0xF8) == 0xF0)) {
            byte second = input.get(input.position() + 1);
            byte third = input.get(input.position() + 2);
            byte fourth = input.get(input.position() + 3);
            int codepoint =
                ((first & 0x07) << 18) | ((second & 0x3F) << 12) | ((third & 0x3F) << 6) | (fourth
                    & 0x3F);
            if (isTrailingByte(second) && isTrailingByte(third) && isTrailingByte(fourth) && (
                codepoint >= 0x10000) && isValidCodepoint(codepoint)) {
                return new UTFCodepoint(codepoint, true, 4);
            }
        }

        // fall-through to returning an invalid codepoint and the replacement codepoint.
        return INVALID;
    }

    /**
     * A UTF codepoint, with the length in UTF-8 bytes.
     * @param codepoint The codepoint.
     * @param valid Was the codepoint parsed from a valid stream?
     * @param length The length of the parsed codepoint. Note if the parsed codepoint was invalid then this length is 1, even though the error codepoint is 3 bytes long.
     */
    public record UTFCodepoint(int codepoint, boolean valid, int length) { }
}
