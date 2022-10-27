/*
 * Copyright (c) 2015-2020, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.interop.onnx;

import ai.onnx.proto.OnnxMl;
import ai.onnxruntime.OnnxModelMetadata;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import com.oracle.labs.mlrg.olcut.provenance.Provenance;
import com.oracle.labs.mlrg.olcut.provenance.io.ProvenanceSerializationException;
import com.oracle.labs.mlrg.olcut.provenance.primitives.LongProvenance;
import com.oracle.labs.mlrg.olcut.provenance.primitives.StringProvenance;
import com.oracle.labs.mlrg.olcut.util.Pair;
import org.tribuo.Example;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.Model;
import org.tribuo.ONNXExportable;
import org.tribuo.Output;
import org.tribuo.OutputFactory;
import org.tribuo.Prediction;
import org.tribuo.impl.ModelDataCarrier;
import org.tribuo.interop.ExternalDatasetProvenance;
import org.tribuo.interop.ExternalModel;
import org.tribuo.interop.ExternalTrainerProvenance;
import org.tribuo.interop.onnx.protos.ONNXExternalModelProto;
import org.tribuo.math.la.SparseVector;
import org.tribuo.protos.ProtoUtil;
import org.tribuo.protos.core.ModelProto;
import org.tribuo.provenance.DatasetProvenance;
import org.tribuo.provenance.ModelProvenance;
import org.tribuo.util.Util;
import org.tribuo.util.onnx.ONNXOperator;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * A Tribuo wrapper around a ONNX model.
 * <p>
 * N.B. ONNX support is experimental, and may change without a major version bump.
 */
public final class ONNXEagerExternalModel<T extends Output<T>> extends ExternalModel<T, OnnxTensor, List<OnnxValue>> implements AutoCloseable {
    private static final long serialVersionUID = 1L;

    private static final Logger logger = Logger.getLogger(ONNXEagerExternalModel.class.getName());

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private transient OrtEnvironment env;

    private transient OrtSession.SessionOptions options;

    private transient List<OrtSession> sessions;

    private final List<ONNXOperator.ONNXOp> ops;

    private final ExampleTransformer featureTransformer;

    private final OutputTransformer<T> outputTransformer;

    private ONNXEagerExternalModel(String name, ModelProvenance provenance,
                                   ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo,
                                   Map<String, Integer> featureMapping,
                                   List<ONNXOperator.ONNXOp> ops,
                                   OrtSession.SessionOptions options,
                                   ExampleTransformer featureTransformer, OutputTransformer<T> outputTransformer) throws OrtException {
        super(name, provenance, featureIDMap, outputIDInfo, outputTransformer.generatesProbabilities(), featureMapping);
        this.options = options;
        this.featureTransformer = featureTransformer;
        this.outputTransformer = outputTransformer;
        this.env = OrtEnvironment.getEnvironment();
        this.ops = ops;
        this.sessions = createSessions(env, ops, options);
    }

    private ONNXEagerExternalModel(String name, ModelProvenance provenance,
                                   ImmutableFeatureMap featureIDMap, ImmutableOutputInfo<T> outputIDInfo,
                                   int[] featureForwardMapping, int[] featureBackwardMapping,
                                   List<ONNXOperator.ONNXOp> ops, OrtSession.SessionOptions options,
                                   ExampleTransformer featureTransformer, OutputTransformer<T> outputTransformer) throws OrtException {
        super(name, provenance, featureIDMap, outputIDInfo, featureForwardMapping, featureBackwardMapping,
                outputTransformer.generatesProbabilities());
        this.options = options;
        this.featureTransformer = featureTransformer;
        this.outputTransformer = outputTransformer;
        this.env = OrtEnvironment.getEnvironment();
        this.ops = ops;
        this.sessions = createSessions(env, ops, options);
    }

    /**
     * Closes the session and rebuilds it using the supplied options.
     * <p>
     * Used to select a different backend, or change the number of inference threads etc.
     *
     * @param newOptions The new session options.
     * @throws OrtException If the model failed to rebuild the session with the supplied options.
     */
    public synchronized void rebuild(OrtSession.SessionOptions newOptions) throws OrtException {
        for (OrtSession s : sessions) {
            s.close();
        }
        if (options != null) {
            options.close();
        }
        options = newOptions;
        sessions = createSessions(env, ops, options);
    }

    private static List<OrtSession> createSessions(OrtEnvironment env, List<ONNXOperator.ONNXOp> ops, OrtSession.SessionOptions options) throws OrtException {
        List<OrtSession> sessions = new ArrayList<>();

        for (ONNXOperator.ONNXOp op : ops) {
            OnnxMl.ModelProto modelProto = op.makeSingleOpModel();
            sessions.add(env.createSession(modelProto.toByteArray(),options));
        }

        return sessions;
    }

    @Override
    protected OnnxTensor convertFeatures(SparseVector input) {
        try {
            return featureTransformer.transform(env, input);
        } catch (OrtException e) {
            throw new IllegalStateException("Failed to construct input OnnxTensor", e);
        }
    }

    @Override
    protected OnnxTensor convertFeaturesList(List<SparseVector> input) {
        try {
            return featureTransformer.transform(env, input);
        } catch (OrtException e) {
            throw new IllegalStateException("Failed to construct input OnnxTensor", e);
        }
    }

    /**
     * Runs the session to make a prediction.
     * <p>
     * Closes the input tensor after the prediction has been made.
     *
     * @param input The input in the external model's format.
     * @return A tensor representing the output.
     */
    @Override
    protected List<OnnxValue> externalPrediction(OnnxTensor input) {
        try {
            // Note the output of the session is closed by the conversion methods, and should not be closed by the result object.
            OrtSession.Result tmpOutputs = null;
            for (OrtSession s : sessions) {
                OnnxTensor o = (OnnxTensor) (tmpOutputs == null ? input : tmpOutputs.get(0));
                OrtSession.Result curOutput = s.run(Map.of("input",o));
                if (tmpOutputs != null) {
                    tmpOutputs.close();
                }
                tmpOutputs = curOutput;
            }
            input.close();
            ArrayList<OnnxValue> outputs = new ArrayList<>();
            for (Map.Entry<String, OnnxValue> v : tmpOutputs) {
                outputs.add(v.getValue());
            }
            return outputs;
        } catch (OrtException e) {
            throw new IllegalStateException("Failed to execute ONNX model", e);
        }
    }

    /**
     * Converts a tensor into a prediction.
     * Closes the output tensor after it's been converted.
     *
     * @param output           The output of the external model.
     * @param numValidFeatures The number of valid features in the input.
     * @param example          The input example, used to construct the Prediction.
     * @return A {@link Prediction} representing this tensor output.
     */
    @Override
    protected Prediction<T> convertOutput(List<OnnxValue> output, int numValidFeatures, Example<T> example) {
        Prediction<T> pred = outputTransformer.transformToPrediction(output, outputIDInfo, numValidFeatures, example);
        OnnxValue.close(output);
        return pred;
    }

    /**
     * Converts a tensor into a prediction.
     * Closes the output tensor after it's been converted.
     *
     * @param output           The output of the external model.
     * @param numValidFeatures An array with the number of valid features in each example.
     * @param examples         The input examples, used to construct the Predictions.
     * @return A list of {@link Prediction} representing this tensor output.
     */
    @Override
    protected List<Prediction<T>> convertOutput(List<OnnxValue> output, int[] numValidFeatures, List<Example<T>> examples) {
        List<Prediction<T>> predictions = outputTransformer.transformToBatchPrediction(output, outputIDInfo, numValidFeatures, examples);
        OnnxValue.close(output);
        return predictions;
    }

    @Override
    public Map<String, List<Pair<String, Double>>> getTopFeatures(int n) {
        return Collections.emptyMap();
    }

    @Override
    protected synchronized Model<T> copy(String newName, ModelProvenance newProvenance) {
        try {
            return new ONNXEagerExternalModel<>(newName, newProvenance, featureIDMap, outputIDInfo,
                    featureForwardMapping, featureBackwardMapping,
                    ops, options, featureTransformer, outputTransformer);
        } catch (OrtException e) {
            throw new IllegalStateException("Failed to copy ONNX model", e);
        }
    }

    @Override
    public void close() {
        if (sessions != null) {
            try {
                for (OrtSession s : sessions) {
                    s.close();
                }
            } catch (OrtException e) {
                logger.log(Level.SEVERE, "Exception thrown when closing sessions", e);
            }
        }
        if (options != null) {
            options.close();
        }
        if (env != null) {
            env.close();
        }
    }

    @Override
    public ModelProto serialize() {
        ModelDataCarrier<T> carrier = createDataCarrier();

        ONNXExternalModelProto.Builder modelBuilder = ONNXExternalModelProto.newBuilder();
        modelBuilder.setMetadata(carrier.serialize());
        modelBuilder.addAllForwardFeatureMapping(Arrays.stream(featureForwardMapping).boxed().collect(Collectors.toList()));
        modelBuilder.addAllBackwardFeatureMapping(Arrays.stream(featureBackwardMapping).boxed().collect(Collectors.toList()));
        modelBuilder.setOutputTransformer(outputTransformer.serialize());
        modelBuilder.setExampleTransformer(featureTransformer.serialize());

        ModelProto.Builder builder = ModelProto.newBuilder();
        builder.setSerializedData(Any.pack(modelBuilder.build()));
        builder.setClassName(ONNXEagerExternalModel.class.getName());
        builder.setVersion(CURRENT_VERSION);

        return builder.build();
    }

    /**
     * Creates an {@code ONNXExternalModel} by loading the model from disk.
     *
     * @param factory            The output factory to use.
     * @param featureMapping     The feature mapping between Tribuo names and ONNX integer ids.
     * @param outputMapping      The output mapping between Tribuo outputs and ONNX integer ids.
     * @param featureTransformer The transformation function for the features.
     * @param outputTransformer  The transformation function for the outputs.
     * @param opts               The session options for the ONNX model.
     * @param ops                The ONNX operations.
     * @param <T>                The type of the output.
     * @return An ONNXExternalModel ready to score new inputs.
     * @throws OrtException If the onnx-runtime native library call failed.
     */
    public static <T extends Output<T>> ONNXEagerExternalModel<T> createOnnxModel(OutputFactory<T> factory,
                                                                                  Map<String, Integer> featureMapping,
                                                                                  Map<T, Integer> outputMapping,
                                                                                  ExampleTransformer featureTransformer,
                                                                                  OutputTransformer<T> outputTransformer,
                                                                                  OrtSession.SessionOptions opts,
                                                                                  List<ONNXOperator.ONNXOp> ops) throws OrtException, MalformedURLException {
        URL provenanceLocation = Paths.get(".").toUri().toURL();
        ImmutableFeatureMap featureMap = ExternalModel.createFeatureMap(featureMapping.keySet());
        ImmutableOutputInfo<T> outputInfo = ExternalModel.createOutputInfo(factory, outputMapping);
        OffsetDateTime now = OffsetDateTime.now();
        ExternalTrainerProvenance trainerProvenance = new ExternalTrainerProvenance(provenanceLocation);
        DatasetProvenance datasetProvenance = new ExternalDatasetProvenance("unknown-external-data", factory, false, featureMapping.size(), outputMapping.size());
        ModelProvenance provenance = new ModelProvenance(ONNXEagerExternalModel.class.getName(), now, datasetProvenance, trainerProvenance);
        return new ONNXEagerExternalModel<>("external-model", provenance, featureMap, outputInfo,
                featureMapping, ops, opts, featureTransformer, outputTransformer);
    }

}
