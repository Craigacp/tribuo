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

public record SPToken(String piece, int id, String surface, int start, int end) {

    SPToken(byte[] piece, int id, byte[] surface, int start, int end) {
        this(UTF8Utils.UTF8.decode(ByteBuffer.wrap(piece)).toString(), id, UTF8Utils.UTF8.decode(ByteBuffer.wrap(surface)).toString(), start, end);
    }

    SPToken(String piece, int id, ByteBuffer surface, int start, int end) {
        this(piece, id, UTF8Utils.UTF8.decode(surface).toString(), start, end);
    }

    SPToken(ByteBuffer piece, int id, ByteBuffer surface, int start, int end) {
        this(UTF8Utils.UTF8.decode(piece).toString(), id, UTF8Utils.UTF8.decode(surface).toString(), start, end);
    }

    SPToken(byte[] piece, int id, ByteBuffer surface, int start, int end) {
        this(UTF8Utils.UTF8.decode(ByteBuffer.wrap(piece)).toString(), id, UTF8Utils.UTF8.decode(surface).toString(), start, end);
    }

    static SPToken merge(SPToken start, SPToken end) {
        if (start.id != end.id) {
            throw new IllegalArgumentException("Can only merge tokens with the same (unknown) id, start.id = " + start.id + ", end.id = " + end.id);
        }
        return new SPToken(start.piece + end.piece, start.id, start.surface + end.surface, start.start, end.end);
    }
}
