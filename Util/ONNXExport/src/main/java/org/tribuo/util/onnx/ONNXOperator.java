/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.util.onnx;

import ai.onnx.proto.OnnxMl;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

import static org.tribuo.util.onnx.ONNXAttribute.VARIADIC_INPUT;

/**
 * An interface for ONNX operators. Usually implemented by an enum representing the opset.
 */
public interface ONNXOperator {

    /**
     * The operator name.
     * @return The name.
     */
    public String getOpName();

    /**
     * The number of inputs.
     * @return The number of inputs.
     */
    public int getNumInputs();

    /**
     * The number of optional inputs.
     * @return The number of optional inputs.
     */
    public int getNumOptionalInputs();

    /**
     * The number of outputs.
     * @return The number of outputs.
     */
    public int getNumOutputs();

    /**
     * The operator attributes.
     * @return The operator attribute map.
     */
    public Map<String,ONNXAttribute> getAttributes();
    
    /**
     * The mandatory attribute names.
     * @return The required attribute names.
     */
    public Set<String> getMandatoryAttributeNames();

    /**
     * Returns the opset version.
     * @return The opset version.
     */
    public int getOpVersion();

    /**
     * Returns the opset domain.
     * <p>
     * May be {@code null} if it is the default ONNX domain;
     * @return The opset domain.
     */
    public String getOpDomain();

    /**
     * Returns the opset proto for these operators.
     * @return The opset proto.
     */
    default public OnnxMl.OperatorSetIdProto opsetProto() {
        return OnnxMl.OperatorSetIdProto.newBuilder().setDomain(getOpDomain()).setVersion(getOpVersion()).build();
    }

    /**
     * Builds this node based on the supplied inputs and output.
     * Throws {@link IllegalArgumentException} if this operator takes more than a single input or output.
     * @param context The onnx context used to ensure this node has a unique name.
     * @param input The name of the input.
     * @param output The name of the output.
     * @return The NodeProto.
     */
    default public OnnxMl.NodeProto build(ONNXContext context, String input, String output) {
        return new ONNXOp(this, input, output).build(context);
    }

    /**
     * Builds this node based on the supplied inputs and output.
     * Throws {@link IllegalArgumentException} if this operator takes more than a single input or output.
     * May throw {@link UnsupportedOperationException} if the attribute type is not supported.
     * @param context The onnx context used to ensure this node has a unique name.
     * @param input The names of the input.
     * @param output The name of the output.
     * @param attributeValues The attribute names and values.
     * @return The NodeProto.
     */
    default public OnnxMl.NodeProto build(ONNXContext context, String input, String output, Map<String,Object> attributeValues) {
        return new ONNXOp(this, input, output, attributeValues).build(context);
    }

    /**
     * Builds this node based on the supplied inputs and output.
     * Throws {@link IllegalArgumentException} if the number of inputs or outputs is wrong.
     * @param context The onnx context used to ensure this node has a unique name.
     * @param inputs The names of the inputs.
     * @param output The name of the output.
     * @return The NodeProto.
     */
    default public OnnxMl.NodeProto build(ONNXContext context, String[] inputs, String output) {
        return new ONNXOp(this, Arrays.asList(inputs), output).build(context);
    }

    /**
     * Builds this node based on the supplied inputs and output.
     * Throws {@link IllegalArgumentException} if the number of inputs, outputs or attributes is wrong.
     * May throw {@link UnsupportedOperationException} if the attribute type is not supported.
     * @param context The onnx context used to ensure this node has a unique name.
     * @param inputs The names of the inputs.
     * @param output The name of the output.
     * @param attributeValues The attribute names and values.
     * @return The NodeProto.
     */
    default public OnnxMl.NodeProto build(ONNXContext context, String[] inputs, String output, Map<String,Object> attributeValues) {
        return new ONNXOp(this, Arrays.asList(inputs),output,attributeValues).build(context);
    }

    /**
     * Builds this node based on the supplied input and outputs.
     * Throws {@link IllegalArgumentException} if the number of inputs or outputs is wrong.
     * @param context The onnx context used to ensure this node has a unique name.
     * @param input The name of the input.
     * @param outputs The names of the outputs.
     * @return The NodeProto.
     */
    default public OnnxMl.NodeProto build(ONNXContext context, String input, String[] outputs) {
        return new ONNXOp(this, input, Arrays.asList(outputs)).build(context);
    }

    /**
     * Builds this node based on the supplied input and outputs.
     * Throws {@link IllegalArgumentException} if the number of inputs, outputs or attributes is wrong.
     * May throw {@link UnsupportedOperationException} if the attribute type is not supported.
     * @param context The onnx context used to ensure this node has a unique name.
     * @param input The name of the input.
     * @param outputs The names of the outputs.
     * @param attributeValues The attribute names and values.
     * @return The NodeProto.
     */
    default public OnnxMl.NodeProto build(ONNXContext context, String input, String[] outputs, Map<String,Object> attributeValues) {
        return new ONNXOp(this, input, Arrays.asList(outputs),attributeValues).build(context);
    }

    /**
     * Builds this node based on the supplied inputs and outputs.
     * Throws {@link IllegalArgumentException} if the number of inputs or outputs is wrong.
     * @param context The onnx context used to ensure this node has a unique name.
     * @param inputs The names of the inputs.
     * @param outputs The names of the outputs.
     * @return The NodeProto.
     */
    default public OnnxMl.NodeProto build(ONNXContext context, String[] inputs, String[] outputs) {
        return new ONNXOp(this, Arrays.asList(inputs), Arrays.asList(outputs)).build(context);
    }

    /**
     * Builds this node based on the supplied inputs and outputs.
     * Throws {@link IllegalArgumentException} if the number of inputs, outputs or attributes is wrong.
     * May throw {@link UnsupportedOperationException} if the attribute type is not supported.
     * @param context The onnx context used to ensure this node has a unique name.
     * @param inputs The names of the inputs.
     * @param outputs The names of the outputs.
     * @param attributeValues The attribute names and values.
     * @return The NodeProto.
     */
    default public OnnxMl.NodeProto build(ONNXContext context, String[] inputs, String[] outputs, Map<String,Object> attributeValues) {
        return new ONNXOp(this, Arrays.asList(inputs), Arrays.asList(outputs), attributeValues).build(context);
    }

    public record ONNXOp(ONNXOperator op, List<String> inputs, List<String> outputs, Map<String,Object> attributeValues) {
        public ONNXOp {
            inputs = Collections.unmodifiableList(inputs);
            outputs = Collections.unmodifiableList(outputs);
            attributeValues = Collections.unmodifiableMap(attributeValues);
        }

        public ONNXOp(ONNXOperator op, String input, String output, Map<String, Object> attributeValues) {
            this(op,List.of(input),List.of(output),attributeValues);
        }
        public ONNXOp(ONNXOperator op, String input, List<String> outputs, Map<String, Object> attributeValues) {
            this(op,List.of(input),outputs,attributeValues);
        }
        public ONNXOp(ONNXOperator op, List<String> inputs, String output, Map<String, Object> attributeValues) {
            this(op,inputs,List.of(output),attributeValues);
        }
        public ONNXOp(ONNXOperator op, String input, String output) {
            this(op,List.of(input),List.of(output),Map.of());
        }
        public ONNXOp(ONNXOperator op, String input, List<String> outputs) {
            this(op,List.of(input),outputs,Map.of());
        }
        public ONNXOp(ONNXOperator op, List<String> inputs, String output) {
            this(op,inputs,List.of(output),Map.of());
        }
        public ONNXOp(ONNXOperator op, List<String> inputs, List<String> outputs) {
            this(op,inputs,outputs,Map.of());
        }

        /**
         * Builds a node from this op record.
         * Throws {@link IllegalArgumentException} if the number of inputs, outputs or attributes is wrong.
         * May throw {@link UnsupportedOperationException} if the attribute type is not supported.
         * @param context The onnx context used to ensure the generated node has a unique name.
         * @return The NodeProto.
         */
        public OnnxMl.NodeProto build(ONNXContext context) {
            int numInputs = op.getNumInputs();
            int numOptionalInputs = op.getNumOptionalInputs();
            int numOutputs = op.getNumOutputs();
            String opName = op.getOpName();
            String domain = op.getOpDomain();
            Map<String, ONNXAttribute> attributes = op.getAttributes();
            Set<String> mandatoryAttributeNames = op.getMandatoryAttributeNames();

            String opStatus = String.format("Building op %s:%s(%d(+%d)) -> %d", domain, opName, numInputs, numOptionalInputs, numOutputs);

            if ((numInputs != VARIADIC_INPUT) && ((inputs.size() < numInputs) || (inputs.size() > numInputs + numOptionalInputs))) {
                throw new IllegalArgumentException(opStatus + ". Expected " + numInputs + " inputs, with " + numOptionalInputs + " optional inputs, but received " + inputs.size());
            } else if ((numInputs == VARIADIC_INPUT) && (inputs.size() == 0)) {
                throw new IllegalArgumentException(opStatus + ". Expected at least one input for variadic input, received zero");
            }
            if (outputs.size() != numOutputs) {
                throw new IllegalArgumentException(opStatus + ". Expected " + numOutputs + " outputs, but received " + outputs.size());
            }
            if (!attributes.keySet().containsAll(attributeValues.keySet())) {
                throw new IllegalArgumentException(opStatus + ". Unexpected attribute found, received " + attributeValues.keySet() + ", expected values from " + attributes.keySet());
            }
            if (!attributeValues.keySet().containsAll(mandatoryAttributeNames)) {
                throw new IllegalArgumentException(opStatus + ". Expected to find all mandatory attributes, received " + attributeValues.keySet() + ", expected " + mandatoryAttributeNames);
            }

            Logger.getLogger("org.tribuo.util.onnx.ONNXOperator").fine(opStatus);
            OnnxMl.NodeProto.Builder nodeBuilder = OnnxMl.NodeProto.newBuilder();
            for (String i : inputs()) {
                nodeBuilder.addInput(i);
            }
            for (String o : outputs()) {
                nodeBuilder.addOutput(o);
            }
            nodeBuilder.setName(context.generateUniqueName(opName));
            nodeBuilder.setOpType(opName);
            if (domain != null) {
                nodeBuilder.setDomain(domain);
            }
            for (Map.Entry<String,Object> e : attributeValues.entrySet()) {
                ONNXAttribute attr = attributes.get(e.getKey());
                nodeBuilder.addAttribute(attr.build(e.getValue()));
            }
            return nodeBuilder.build();
        }

        public OnnxMl.ModelProto makeSingleOpModel() {
            ONNXContext context = new ONNXContext();
            OnnxMl.ValueInfoProto inputValue = OnnxMl.ValueInfoProto.newBuilder().setName(inputs().get(0)).build();
            context.protoBuilder.addInput(inputValue);
            OnnxMl.ValueInfoProto outputValue = OnnxMl.ValueInfoProto.newBuilder().setName(outputs().get(0)).build();
            context.protoBuilder.addOutput(outputValue);
            OnnxMl.NodeProto opNode = build(context);
            context.protoBuilder.addNode(opNode);
            return OnnxMl.ModelProto.newBuilder()
                    .setGraph(context.buildGraph())
                    .setDomain("org.tribuo.onnx.test")
                    .setProducerName("Tribuo")
                    .setProducerVersion("5.0.0")
                    .setModelVersion(0)
                    .addOpsetImport(ONNXOperators.getOpsetProto())
                    .setIrVersion(6)
                    .setDocString("eager-test")
                    .build();
        }
    }
}
