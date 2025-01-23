/*
 * Copyright (c) 2015, 2025, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.math;

import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseSparseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.Matrix;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.la.Tensor;
import org.tribuo.math.protos.LinearParametersProto;
import org.tribuo.math.protos.ParametersProto;
import org.tribuo.math.protos.TensorProto;
import org.tribuo.math.util.HeapMerger;
import org.tribuo.math.util.Merger;
import org.tribuo.protos.ProtoSerializableClass;
import org.tribuo.protos.ProtoSerializableField;
import org.tribuo.protos.ProtoUtil;

import java.util.Objects;

/**
 * A {@link Parameters} for producing linear models.
 */
@ProtoSerializableClass(version = LinearParameters.CURRENT_VERSION, serializedDataClass = LinearParametersProto.class)
public class LinearParameters implements FeedForwardParameters {
    private static final long serialVersionUID = 1L;

    /**
     * Protobuf serialization version.
     */
    public static final int CURRENT_VERSION = 0;

    private static final Merger merger = new HeapMerger();

    /**
     * A single element array, containing {@link #weightMatrix}.
     */
    private Tensor[] weights;

    /**
     * Last row in this DenseMatrix is the bias, added by
     * calling SparseVector.createSparseVector(example,featureInfo,true);
     */
    @ProtoSerializableField
    private DenseMatrix weightMatrix;

    /**
     * Constructor. The number of features and the number of outputs must be fixed and known in advance.
     * @param numFeatures The number of features in the training dataset (excluding the bias).
     * @param numLabels The number of outputs in the training dataset.
     */
    public LinearParameters(int numFeatures, int numLabels) {
        weights = new Tensor[1];
        weightMatrix = new DenseMatrix(numLabels,numFeatures);
        weights[0] = weightMatrix;
    }

    /**
     * Constructs a LinearParameters wrapped around a weight matrix.
     * <p>
     * Used for serialization compatibility with Tribuo 4.0.
     * @param weightMatrix The weight matrix to wrap.
     */
    public LinearParameters(DenseMatrix weightMatrix) {
        this.weightMatrix = weightMatrix;
        this.weights = new Tensor[1];
        weights[0] = weightMatrix;
    }

    /**
     * Deserialization factory.
     * @param version The serialized object version.
     * @param className The class name.
     * @param message The serialized data.
     * @throws InvalidProtocolBufferException If the protobuf could not be parsed from the {@code message}.
     * @return The deserialized object.
     */
    public static LinearParameters deserializeFromProto(int version, String className, Any message) throws InvalidProtocolBufferException {
        if (version < 0 || version > CURRENT_VERSION) {
            throw new IllegalArgumentException("Unknown version " + version + ", this class supports at most version " + CURRENT_VERSION);
        }
        LinearParametersProto proto = message.unpack(LinearParametersProto.class);
        TensorProto tensorProto = proto.getWeightMatrix();
        Tensor tensor = ProtoUtil.deserialize(tensorProto);
        if (tensor instanceof DenseMatrix) {
            return new LinearParameters((DenseMatrix)tensor);
        } else {
            throw new IllegalStateException("Invalid protobuf, found a " + tensor.getClass().getSimpleName() + " when expecting a dense matrix.");
        }
    }

    @Override
    public ParametersProto serialize() {
        return ProtoUtil.serialize(this);
    }

    /**
     * Generates an unnormalised prediction by leftMultiply'ing the weights with the incoming features.
     * @param example A feature vector
     * @return A {@link DenseVector} containing a score for each label.
     */
    @Override
    public DenseVector predict(SGDVector example) {
        return weightMatrix.leftMultiply(example);
    }

    /**
     * Generates an unnormalised prediction by matrix multiplying the weights with the incoming feature batch.
     * @param batch A feature matrix
     * @return A {@link DenseMatrix} containing a score for each output dimension and each batch member.
     */
    @Override
    public DenseMatrix predict(Matrix batch) {
        return (DenseMatrix) batch.matrixMultiply(weightMatrix, false, true);
    }

    /**
     * Generate the gradients for a particular feature vector given
     * the loss and the per output gradients.
     * <p>
     * This method returns a single element {@link Tensor} array.
     * @param score The Pair returned by the objective.
     * @param features The feature vector.
     * @return A {@link Tensor} array with a single {@link Matrix} containing all gradients.
     */
    @Override
    public Tensor[] gradients(LossAndGrad score, SGDVector features) {
        Matrix gradient = score.gradient().outer(features);
        return new Tensor[]{gradient};
    }

    @Override
    public Tensor[] gradients(BatchLossAndGrad score, Matrix featureBatch) {
        Matrix gradient = score.gradient().matrixMultiply(featureBatch, true, false);
        return new Tensor[]{gradient};
    }

    /**
     * This returns a {@link DenseMatrix} the same size as the Parameters.
     * @return A {@link Tensor} array containing a single {@link DenseMatrix}.
     */
    @Override
    public Tensor[] getEmptyCopy() {
        DenseMatrix matrix = new DenseMatrix(weightMatrix.getDimension1Size(),weightMatrix.getDimension2Size());
        Tensor[] output = new Tensor[1];
        output[0] = matrix;
        return output;
    }

    @Override
    public Tensor[] get() {
        return weights;
    }

    /**
     * Returns the weight matrix.
     * @return The weight matrix.
     */
    public DenseMatrix getWeightMatrix() {
        return weightMatrix;
    }

    @Override
    public void set(Tensor[] newWeights) {
        if (newWeights.length == weights.length) {
            weights = newWeights;
            weightMatrix = (DenseMatrix) weights[0];
        }
    }

    @Override
    public void update(Tensor[] gradients) {
        for (int i = 0; i < gradients.length; i++) {
            weights[i].intersectAndAddInPlace(gradients[i]);
        }
    }

    @Override
    public Tensor[] merge(Tensor[][] gradients, int size) {
        if (gradients[0][0] instanceof DenseMatrix) {
            for (int i = 1; i < size; i++) {
                gradients[0][0].intersectAndAddInPlace(gradients[i][0]);
            }
            return new Tensor[]{gradients[0][0]};
        } else if (gradients[0][0] instanceof DenseSparseMatrix) {
            DenseSparseMatrix[] updates = new DenseSparseMatrix[size];
            for (int j = 0; j < updates.length; j++) {
                updates[j] = (DenseSparseMatrix) gradients[j][0];
            }

            DenseSparseMatrix update = merger.merge(updates);

            return new Tensor[]{update};
        } else {
            throw new IllegalStateException("Unexpected gradient type, expected DenseMatrix or DenseSparseMatrix, received " + gradients[0][0].getClass().getName());
        }
    }

    @Override
    public LinearParameters copy() {
        return new LinearParameters(weightMatrix.copy());
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        LinearParameters that = (LinearParameters) o;
        return weightMatrix.equals(that.weightMatrix);
    }

    @Override
    public int hashCode() {
        return Objects.hash(weightMatrix);
    }
}
