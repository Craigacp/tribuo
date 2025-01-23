/*
 * Copyright (c) 2020, 2025, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.multilabel.sgd.objectives;

import com.oracle.labs.mlrg.olcut.provenance.ConfiguredObjectProvenance;
import com.oracle.labs.mlrg.olcut.provenance.impl.ConfiguredObjectProvenanceImpl;
import org.tribuo.math.Parameters;
import org.tribuo.math.la.DenseMatrix;
import org.tribuo.math.la.DenseSparseMatrix;
import org.tribuo.math.la.DenseVector;
import org.tribuo.math.la.Matrix;
import org.tribuo.math.la.SparseVector;
import org.tribuo.math.util.SigmoidNormalizer;
import org.tribuo.multilabel.sgd.MultiLabelObjective;
import org.tribuo.math.la.SGDVector;
import org.tribuo.math.util.VectorNormalizer;

/**
 * A multilabel version of binary cross entropy loss which expects logits.
 * <p>
 * Generates a probabilistic model, and uses a {@link SigmoidNormalizer}.
 */
public final class BinaryCrossEntropy implements MultiLabelObjective {

    private static final VectorNormalizer normalizer = new SigmoidNormalizer();

    /**
     * Constructs a BinaryCrossEntropy objective.
     */
    public BinaryCrossEntropy() {}

    /**
     * Returns a {@link org.tribuo.math.Parameters.LossAndGrad} containing the loss and per label gradients.
     * <p>
     * The prediction vector is transformed to produce the per label gradient and returned.
     * @param truth The true label id
     * @param prediction The prediction for each label id
     * @return The score and per label gradient.
     */
    @Override
    public Parameters.LossAndGrad lossAndGradient(SGDVector truth, SGDVector prediction) {
        DenseVector labels, densePred;
        if (truth instanceof SparseVector) {
            labels = ((SparseVector) truth).densify();
        } else {
            labels = (DenseVector) truth;
        }
        if (prediction instanceof SparseVector) {
            densePred = ((SparseVector) prediction).densify();
        } else {
            densePred = (DenseVector) prediction;
        }

        double loss = 0.0;
        for (int i = 0; i < prediction.size(); i++) {
            double label = labels.get(i);
            double pred = densePred.get(i);
            double yhat = SigmoidNormalizer.sigmoid(pred);
            // numerically stable form of loss computation
            loss += Math.max(pred, 0) - (pred * label) + Math.log1p(Math.exp(-Math.abs(pred)));
            densePred.set(i,-(yhat - label));
        }
        return new Parameters.LossAndGrad(loss,densePred);
    }

    /**
     * Returns a {@link org.tribuo.math.Parameters.BatchLossAndGrad} containing the loss and per label gradients.
     * <p>
     * The prediction vector is transformed to produce the per label gradient and returned.
     * @param truth The true label id
     * @param prediction The prediction for each label id
     * @return The score and per label gradient.
     */
    @Override
    public Parameters.BatchLossAndGrad batchLossAndGradient(Matrix truth, DenseMatrix prediction) {
        DenseMatrix labels;
        if (truth instanceof DenseSparseMatrix) {
            labels = ((DenseSparseMatrix) truth).densify();
        } else {
            labels = (DenseMatrix) truth;
        }

        double[] loss = new double[prediction.getDimension1Size()];
        for (int i = 0; i < prediction.getDimension1Size(); i++) {
            for (int j = 0; j < prediction.getDimension2Size(); j++) {
                double label = labels.get(i,j);
                double pred = prediction.get(i,j);
                double yhat = SigmoidNormalizer.sigmoid(pred);
                // numerically stable form of loss computation
                loss[i] += Math.max(pred, 0) - (pred * label) + Math.log1p(Math.exp(-Math.abs(pred)));
                prediction.set(i, j, -(yhat - label));
            }
        }
        return new Parameters.BatchLossAndGrad(loss,prediction);
    }

    @Override
    public double loss(SGDVector truth, SGDVector prediction) {
        DenseVector labels, densePred;
        if (truth instanceof SparseVector) {
            labels = ((SparseVector) truth).densify();
        } else {
            labels = (DenseVector) truth;
        }
        if (prediction instanceof SparseVector) {
            densePred = ((SparseVector) prediction).densify();
        } else {
            densePred = (DenseVector) prediction;
        }

        double loss = 0.0;
        for (int i = 0; i < prediction.size(); i++) {
            double label = labels.get(i);
            double pred = densePred.get(i);
            // numerically stable form of loss computation
            loss += Math.max(pred, 0) - (pred * label) + Math.log1p(Math.exp(-Math.abs(pred)));
        }
        return loss;
    }

    @Override
    public double[] batchLoss(Matrix truth, DenseMatrix prediction) {
        DenseMatrix labels;
        if (truth instanceof DenseSparseMatrix) {
            labels = ((DenseSparseMatrix) truth).densify();
        } else {
            labels = (DenseMatrix) truth;
        }

        double[] loss = new double[prediction.getDimension1Size()];
        for (int i = 0; i < prediction.getDimension1Size(); i++) {
            for (int j = 0; j < prediction.getDimension2Size(); j++) {
                double label = labels.get(i,j);
                double pred = prediction.get(i,j);
                // numerically stable form of loss computation
                loss[i] += Math.max(pred, 0) - (pred * label) + Math.log1p(Math.exp(-Math.abs(pred)));
            }
        }
        return loss;
    }

    @Override
    public VectorNormalizer getNormalizer() {
        return normalizer;
    }

    /**
     * Returns true.
     * @return True.
     */
    @Override
    public boolean isProbabilistic() {
        return true;
    }

    @Override
    public double threshold() {
        return 0.5;
    }

    @Override
    public String toString() {
        return "BinaryCrossEntropy";
    }

    @Override
    public ConfiguredObjectProvenance getProvenance() {
        return new ConfiguredObjectProvenanceImpl(this,"MultiLabelObjective");
    }
}
