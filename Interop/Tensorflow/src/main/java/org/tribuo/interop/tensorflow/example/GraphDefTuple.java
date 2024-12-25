/*
 * Copyright (c) 2021, 2024, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.interop.tensorflow.example;

import org.tensorflow.proto.GraphDef;

/**
 * A tuple containing a graph def protobuf along with the relevant operation names.
 *
 * @param graphDef   The graph definition protobuf.
 * @param inputName  Name of the input operation.
 * @param outputName Name of the output operation.
 */
public record GraphDefTuple(GraphDef graphDef, String inputName, String outputName) { }
