package weka.classifiers.trees;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.RandomizableClassifier;
import weka.core.*;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.stream.IntStream;

/**
 * WEKA classifiers (and this includes learning algorithms that build regression models!)
 * should simply extend AbstractClassifier.
 */
public class XGBoostTree extends RandomizableClassifier implements WeightedInstancesHandler {

    public double LAMBDA = 1;

    /** A possible way to represent the tree structure using Java records. */
    private interface Node { }
    private record InternalNode(Attribute attribute, double splitPoint, Node leftSuccessor, Node rightSuccessor)
            implements Node, Serializable { }
    private record LeafNode(double prediction) implements Node, Serializable { }

    /** The root node of the decision tree. */
    private Node rootNode = null;

    /** The training instances. */
    private Instances data;

    /** A class for objects that hold a split specification, including the quality of the split. */
    private class SplitSpecification {
        private final Attribute attribute; private double splitPoint; private double splitQuality;
        private SplitSpecification(Attribute attribute, double splitQuality, double splitPoint) {
            this.attribute = attribute; this.splitQuality = splitQuality; this.splitPoint = splitPoint;
        }
    }

    /**
     * A class for objects that contain the sufficient statistics required to measure split quality,
     * These statistics are sufficient to compute the sum of squared deviations from the mean.
     */
    private class SufficientStatistics {
        private int n = 0; private double sum = 0.0; private double sumOfSquares = 0.0;
        private SufficientStatistics(int n, double sum, double sumOfSquares) {
            this.n = n; this.sum = sum; this.sumOfSquares = sumOfSquares;
        }
        private void updateStats(double classValue, boolean add) {
            n = (add) ? n + 1 : n - 1;
            sum = (add) ? sum + classValue : sum - classValue;
            sumOfSquares = (add) ? sumOfSquares + classValue * classValue : sumOfSquares - classValue * classValue;
        }
    }
    private class SufficientStatisticsXg {
        private int n = 0; private double sum = 0.0; private double sumOfGrad = 0.0; private double sumOfHess = 0.0;
        private SufficientStatisticsXg(int n, double sum, double sumOfGrad, double sumOfHess) {
            this.n = n; this.sum = sum; this.sumOfGrad = sumOfGrad; this.sumOfHess = sumOfHess;
        }
        private void updateStats(double grad, double hess, boolean add) {
            n = (add) ? n + 1 : n - 1;
            sumOfGrad = (add) ? sumOfGrad + grad : sumOfGrad - grad;
            sumOfHess = (add) ? sumOfHess + hess : sumOfHess - hess;
        }
    }

    /** Computes the sum of squared deviations from the mean based on the sufficient statistics provided. */
    private double sumOfSquaredDeviationsFromTheMean(SufficientStatistics stats) {
        return stats.sumOfSquares - stats.sum * stats.sum / stats.n;
    }

    /** Computes w* */
    private double calcWeight(double sumG, double sumH){
        return (sumG / (sumH + LAMBDA) * (-1));
    }

    /**
     * Computes the reduction in the sum of squared errors based on the sufficient statistics provided. The
     * initialSufficientStatistics are the sufficient statistics based on the data before it is split,
     * statsLeft are the sufficient statistics for the left branch, and statsRight are the sufficient
     * statistics for the right branch.
     */
    private double splitQuality(SufficientStatistics initialSufficientStatistics,
                                SufficientStatistics statsLeft, SufficientStatistics statsRight) {
        return sumOfSquaredDeviationsFromTheMean(initialSufficientStatistics) -
                (sumOfSquaredDeviationsFromTheMean(statsLeft) + sumOfSquaredDeviationsFromTheMean(statsRight));
    }
    private double splitQualityXg(SufficientStatisticsXg initialSufficientStatistics,
                                  SufficientStatisticsXg statsLeft, SufficientStatisticsXg statsRight) {
        return (statsLeft.sumOfGrad * statsLeft.sumOfGrad / (statsLeft.sumOfHess * statsLeft.sumOfHess + LAMBDA)) +
                (statsRight.sumOfGrad * statsRight.sumOfGrad / (statsRight.sumOfHess * statsRight.sumOfHess + LAMBDA)) -
                ((statsLeft.sumOfGrad + statsRight.sumOfGrad) * (statsLeft.sumOfGrad + statsRight.sumOfGrad) /
                        ((statsLeft.sumOfHess + statsRight.sumOfHess) + LAMBDA));
    }

    /**
     * Finds the best split point and returns the corresponding split specification object. The given indices
     * define the subset of the training set for which the split is to be found. The initialStats are the sufficient
     * statistics before the data is split.
     */
    private SplitSpecification findBestSplitPoint(int[] indices, Attribute attribute,
                                                  SufficientStatistics initialStats) {
        var statsLeft = new SufficientStatistics(initialStats.n, initialStats.sum, initialStats.sumOfSquares);
        var statsRight = new SufficientStatistics(0, 0.0, 0.0);
        var splitSpecification = new SplitSpecification(attribute, 0.0, Double.NEGATIVE_INFINITY);
        var previousValue = Double.NEGATIVE_INFINITY;
//        var xx = Arrays.stream(indices).mapToDouble(x -> data.instance(x).value(attribute)).toArray();
//        var xxx = Utils.sortWithNoMissingValues(Arrays.stream(indices).mapToDouble(x ->
//                data.instance(x).value(attribute)).toArray());
//        var xxxx = Arrays.stream(Utils.sortWithNoMissingValues(Arrays.stream(indices).mapToDouble(x -> data.instance(x).value(attribute)).toArray())).map(x -> indices[x]).toArray();
        for (int i : Arrays.stream(Utils.sortWithNoMissingValues(Arrays.stream(indices).mapToDouble(x -> data.instance(x).value(attribute)).toArray())).map(x -> indices[x]).toArray()) {
            Instance instance = data.instance(i);
            if (instance.value(attribute) > previousValue) {
                var splitQuality = splitQuality(initialStats, statsLeft, statsRight);
                if (splitQuality > splitSpecification.splitQuality) {
                    splitSpecification.splitQuality = splitQuality;
                    splitSpecification.splitPoint = (instance.value(attribute) + previousValue) / 2.0;
                }

                previousValue = instance.value(attribute);
            }
            statsLeft.updateStats(instance.classValue(), false);
            statsRight.updateStats(instance.classValue(), true);
        }
        return splitSpecification;
    }
    private SplitSpecification findBestSplitPointXg(int[] indices, Attribute attribute,
                                                  SufficientStatisticsXg initialStats) {
        double gSumLeft = 0; double gSumRight = 0; double hSumLeft = 0; double hSumRight = 0; double gain = 0;
        var statsLeft = new SufficientStatisticsXg(initialStats.n, initialStats.sum, initialStats.sumOfGrad, initialStats.sumOfHess);
        var statsRight = new SufficientStatisticsXg(0, 0.0, 0.0, 0.0);
        var splitSpecification = new SplitSpecification(attribute, 0.0, Double.NEGATIVE_INFINITY);
        double prevValue = Double.NEGATIVE_INFINITY;
        for(int i : Arrays.stream(Utils.sortWithNoMissingValues(Arrays.stream(indices).mapToDouble(x ->
                data.instance(x).value(attribute)).toArray())).map(x -> indices[x]).toArray()){
            Instance instance = data.instance(i);
            if(instance.value(attribute) > prevValue) {
                var splitQuality = splitQualityXg()
            }
        }
        return  null;
    }

    /**
     * Recursively grows a tree for a given set of data.
     */
    private Node makeTree(int[] indices) {
//        var stats = new SufficientStatistics(0, 0.0, 0.0);
        var stats = new SufficientStatisticsXg(0, 0.0, 0.0, 0.0);
//        for (int i : indices) { stats.updateStats(data.instance(i).classValue(),true); }
        for (int i : indices) {
//            stats.updateStats(data.instance(i).classValue(),true);
            stats.updateStats(data.instance(i).classValue(),data.instance(i).weight(), true);
        }
//        if (stats.n <= 1) { return new LeafNode(stats.sum / stats.n); }
        if (stats.n <= 1) { return new LeafNode(this.calcWeight(stats.sumOfGrad, stats.sumOfHess)); }
        var bestSplitSpecification = new SplitSpecification(null, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY);
        for (Attribute attribute : Collections.list(data.enumerateAttributes())) {
            var SplitSpecification = findBestSplitPointXg(indices, attribute, stats);
            if (SplitSpecification.splitQuality > bestSplitSpecification.splitQuality) {
                bestSplitSpecification = SplitSpecification;
            }
        }
        if (bestSplitSpecification.splitQuality < 1E-6) {
            return new LeafNode(stats.sum / stats.n);
        } else {
            var leftSubset = new ArrayList<Integer>(indices.length);
            var rightSubset = new ArrayList<Integer>(indices.length);
            for (int i : indices) {
                if (data.instance(i).value(bestSplitSpecification.attribute) < bestSplitSpecification.splitPoint) {
                    leftSubset.add(i);
                } else {
                    rightSubset.add(i);
                }
            }
            return new InternalNode(bestSplitSpecification.attribute, bestSplitSpecification.splitPoint,
                    makeTree(leftSubset.stream().mapToInt(Integer::intValue).toArray()),
                    makeTree(rightSubset.stream().mapToInt(Integer::intValue).toArray()));
        }
    }

    /** Returns the capabilities of the classifier: numeric predictors and numeric target. */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        return result;
    }

    /**
     * Builds the tree classifier by making a shallow copy of the training data and calling the
     * recursive makeTree(int[]) method.
     */
    public void buildClassifier(Instances trainingData) throws Exception {
        // First, use the capabilities to check whether the learning algorithm can handle the data.
        getCapabilities().testWithFail(trainingData);
        this.data = new Instances(trainingData);
        rootNode = makeTree(IntStream.range(0, this.data.numInstances()).toArray());
    }

    /** Recursive method for obtaining a prediction from the tree attached to the node provided. */
    private double makePrediction(Node node, Instance instance) {
        if (node instanceof LeafNode) {
            return ((LeafNode) node).prediction;
        } else if (node instanceof InternalNode) {
            if (instance.value(((InternalNode) node).attribute) < ((InternalNode) node).splitPoint) {
                return makePrediction(((InternalNode) node).leftSuccessor, instance);
            } else {
                return makePrediction(((InternalNode) node).rightSuccessor, instance);
            }
        }
        return Utils.missingValue(); // This should never happen
    }

    /**
     * Provides a prediction for the current instance by calling the recursive makePrediction(Node, Instance) method.
     */
    public double classifyInstance(Instance instance) {
        return makePrediction(rootNode, instance);
    }

    /** Recursively produces the string representation of a branch in the tree. */
    private void branchToString(StringBuffer sb, boolean left, int level, InternalNode node) {
        sb.append("\n");
        for (int j = 0; j < level; j++) { sb.append("|   "); }
        sb.append(node.attribute.name() + (left ? " < " : " >= ") + Utils.doubleToString(node.splitPoint, getNumDecimalPlaces()));
        toString(sb, level + 1, left ? node.leftSuccessor : node.rightSuccessor);
    }

    /**
     * Recursively produces a string representation of a subtree by calling the branchToString(StringBuffer, int,
     * Node) method for both branches, unless we are at a leaf.
     */
    private void toString(StringBuffer sb, int level, Node node) {
        if (node instanceof LeafNode) {
            sb.append(": " + ((LeafNode) node).prediction);
        } else {
            branchToString(sb, true, level, (InternalNode) node);
            branchToString(sb, false, level, (InternalNode) node);
        }
    }

    /**
     * Returns a string representation of the tree by calling the recursive toString(StringBuffer, int, Node) method.
     */
    public String toString() {
        if (rootNode == null) {
            return "No model has been built yet.";
        }
        StringBuffer sb = new StringBuffer();
        toString(sb, 0, rootNode);
        return sb.toString();
    }

    /** The main method for running this classifier from a command-line interface. */
    public static void main(String[] options) {
        runClassifier(new XGBoostTree(), options);
    }
}