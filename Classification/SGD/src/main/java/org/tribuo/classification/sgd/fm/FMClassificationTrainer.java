/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
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

package org.tribuo.classification.sgd.fm;

import com.oracle.labs.mlrg.olcut.config.Config;
import org.tribuo.ImmutableFeatureMap;
import org.tribuo.ImmutableOutputInfo;
import org.tribuo.classification.Label;
import org.tribuo.classification.sgd.LabelObjective;
import org.tribuo.classification.sgd.objectives.LogMulticlass;
import org.tribuo.common.sgd.AbstractFMTrainer;
import org.tribuo.common.sgd.FMParameters;
import org.tribuo.common.sgd.SGDObjective;
import org.tribuo.math.StochasticGradientOptimiser;
import org.tribuo.provenance.ModelProvenance;

import java.util.logging.Logger;

/**
 * A trainer for a classification factorization machine using SGD.
 * <p>
 * See:
 * <pre>
 * Rendle, S.
 * Factorization machines.
 * 2010 IEEE International Conference on Data Mining
 * </pre>
 */
public class FMClassificationTrainer extends AbstractFMTrainer<Label, Integer, FMClassificationModel, int[]> {
    private static final Logger logger = Logger.getLogger(FMClassificationTrainer.class.getName());

    @Config(description = "The classification objective function to use.")
    private LabelObjective objective = new LogMulticlass();

    /**
     * Constructs an SGD trainer for a factorization machine.
     *
     * @param objective         The objective function to optimise.
     * @param optimiser         The gradient optimiser to use.
     * @param epochs            The number of epochs (complete passes through the training data).
     * @param loggingInterval   Log the loss after this many iterations. If -1 don't log anything.
     * @param minibatchSize     The size of any minibatches.
     * @param seed              A seed for the random number generator, used to shuffle the examples before each epoch.
     * @param factorizedDimSize Size of the factorized feature representation.
     * @param variance          The variance of the initializer.
     */
    public FMClassificationTrainer(LabelObjective objective, StochasticGradientOptimiser optimiser, int epochs,
                                   int loggingInterval, int minibatchSize, long seed,
                                   int factorizedDimSize, double variance) {
        super(optimiser, epochs, loggingInterval, minibatchSize, seed, factorizedDimSize, variance);
        this.objective = objective;
    }

    /**
     * Constructs an SGD trainer for a factorization machine.
     * <p>
     * Sets the minibatch size to 1.
     *
     * @param objective         The objective function to optimise.
     * @param optimiser         The gradient optimiser to use.
     * @param epochs            The number of epochs (complete passes through the training data).
     * @param loggingInterval   Log the loss after this many iterations. If -1 don't log anything.
     * @param seed              A seed for the random number generator, used to shuffle the examples before each epoch.
     * @param factorizedDimSize Size of the factorized feature representation.
     * @param variance          The variance of the initializer.
     */
    public FMClassificationTrainer(LabelObjective objective, StochasticGradientOptimiser optimiser, int epochs,
                                   int loggingInterval, long seed,
                                   int factorizedDimSize, double variance) {
        this(objective, optimiser, epochs, loggingInterval, 1, seed, factorizedDimSize, variance);
    }

    /**
     * Constructs an SGD trainer for a factorization machine.
     * <p>
     * Sets the minibatch size to 1 and the logging interval to 1000.
     *
     * @param objective         The objective function to optimise.
     * @param optimiser         The gradient optimiser to use.
     * @param epochs            The number of epochs (complete passes through the training data).
     * @param seed              A seed for the random number generator, used to shuffle the examples before each epoch.
     * @param factorizedDimSize Size of the factorized feature representation.
     * @param variance          The variance of the initializer.
     */
    public FMClassificationTrainer(LabelObjective objective, StochasticGradientOptimiser optimiser, int epochs,
                                   long seed, int factorizedDimSize, double variance) {
        this(objective, optimiser, epochs, 1000, 1, seed, factorizedDimSize, variance);
    }

    /**
     * For olcut.
     */
    private FMClassificationTrainer() {
        super();
    }

    @Override
    protected Integer[] createTargetArray(int size) {
        return new Integer[size];
    }

    @Override
    protected Integer getTarget(ImmutableOutputInfo<Label> outputInfo, Label output) {
        return outputInfo.getID(output);
    }

    @Override
    protected LabelObjective getObjective() {
        return objective;
    }

    @Override
    protected int[] getTargetBatch(Integer[] outputs, int start, int size) {
        int[] output = new int[size];
        for (int i = start; i < start+size; i++) {
            output[i - start] = outputs[i];
        }
        return output;
    }

    @Override
    protected FMClassificationModel createModel(String name, ModelProvenance provenance, ImmutableFeatureMap featureMap, ImmutableOutputInfo<Label> outputInfo, FMParameters parameters) {
        return new FMClassificationModel(name, provenance, featureMap, outputInfo, parameters, objective.getNormalizer(), objective.isProbabilistic());
    }

    @Override
    protected String getModelClassName() {
        return FMClassificationModel.class.getName();
    }

    @Override
    public String toString() {
        return "FMClassificationTrainer(objective=" + objective.toString() + ",optimiser=" + optimiser.toString() +
                ",epochs=" + epochs + ",minibatchSize=" + minibatchSize + ",seed=" + seed +
                ",factorizedDimSize=" + factorizedDimSize + ",variance=" + variance +
                ")";
    }
}
