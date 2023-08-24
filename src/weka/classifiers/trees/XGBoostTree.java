package weka.classifiers.trees;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.RandomizableClassifier;
import weka.core.*;

import java.io.Serializable;
import java.util.*;
import java.util.stream.IntStream;

/**
 * WEKA classifiers (and this includes learning algorithms that build regression models!)
 * should simply extend AbstractClassifier.
 */
public class XGBoostTree extends RandomizableClassifier implements WeightedInstancesHandler {

    /** The hyperparameters for an XGBoost tree. */
    private double eta = 0.3;
    @OptionMetadata(displayName = "eta", description = "eta",
            commandLineParamName = "eta", commandLineParamSynopsis = "-eta <double>", displayOrder = 1)
    public void setEta(double e) { eta = e; } public double getEta() {return eta; }

    private double lambda = 1.0;
    @OptionMetadata(displayName = "lambda", description = "lambda",
            commandLineParamName = "lambda", commandLineParamSynopsis = "-lambda <double>", displayOrder = 2)
    public void setLambda(double l) { lambda = l; } public double getLambda() {return lambda; }

    private double gamma = 1.0;
    @OptionMetadata(displayName = "gamma", description = "gamma",
            commandLineParamName = "gamma", commandLineParamSynopsis = "-gamma <double>", displayOrder = 3)
    public void setGamma(double l) { gamma = l; } public double getGamma() {return gamma; }

    private double subsample = 0.5;
    @OptionMetadata(displayName = "subsample", description = "subsample",
            commandLineParamName = "subsample", commandLineParamSynopsis = "-subsample <double>", displayOrder = 4)
    public void setSubsample(double s) { subsample = s; } public double getSubsample() {return subsample; }

    private double colsample_bynode = 0.5;
    @OptionMetadata(displayName = "colsample_bynode", description = "colsample_bynode",
            commandLineParamName = "colsample_bynode", commandLineParamSynopsis = "-colsample_bynode <double>", displayOrder = 5)
    public void setColSampleByNode(double c) { colsample_bynode = c; } public double getColSampleByNode() {return colsample_bynode; }

    private int max_depth = 6;
    @OptionMetadata(displayName = "max_depth", description = "max_depth",
            commandLineParamName = "max_depth", commandLineParamSynopsis = "-max_depth <int>", displayOrder = 6)
    public void setMaxDepth(int m) { max_depth = m; } public int getMaxDepth() {return max_depth; }

    private double min_child_weight = 1.0;
    @OptionMetadata(displayName = "min_child_weight", description = "min_child_weight",
            commandLineParamName = "min_child_weight", commandLineParamSynopsis = "-min_child_weight <double>", displayOrder = 7)
    public void setMinChildWeight(double w) { min_child_weight = w; } public double getMinChildWeight() {return min_child_weight; }

    /**
     * 随机数seed是怎么回事
     * 参数写好
     * experimenter
     * significance
     * level at the default value 0.05？？？
     * runable of code
     * 在特定参数下的表现实验比较*/

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

    /** Computes w* */
    private double calcWeight(double sumG, double sumH){
//        double update = (sumG / (sumH + lambda) * (-1) * eta);
        double update = (sumG / (sumH + lambda) * eta);
        return update;
    }

    /**
     * Computes the reduction in the sum of squared errors based on the sufficient statistics provided. The
     * initialSufficientStatistics are the sufficient statistics based on the data before it is split,
     * statsLeft are the sufficient statistics for the left branch, and statsRight are the sufficient
     * statistics for the right branch.
     */
    private double splitQualityXg(SufficientStatisticsXg initialSufficientStatistics,
                                  SufficientStatisticsXg statsLeft, SufficientStatisticsXg statsRight) {
//        return (statsLeft.sumOfGrad * statsLeft.sumOfGrad / (statsLeft.sumOfHess + lambda)) +
//                (statsRight.sumOfGrad * statsRight.sumOfGrad / (statsRight.sumOfHess + lambda)) -
//                ((initialSufficientStatistics.sumOfGrad * initialSufficientStatistics.sumOfGrad) /
//                        (initialSufficientStatistics.sumOfHess + lambda));
        double gleft = statsLeft.sumOfGrad;
        double gright = statsRight.sumOfGrad;
        double hleft = statsLeft.sumOfHess;
        double hright = statsRight.sumOfHess;
        double posgainl = (gleft * gleft) / (hleft  + lambda);
        double postgainr = (gright * gright) / (hright + lambda);
        double pregain = Math.pow((gleft + gright), 2) / (hleft + hright + lambda);
        double gain  = 0.5 * (posgainl + postgainr - pregain);
        return gain;
    }

    /**
     * Finds the best split point and returns the corresponding split specification object. The given indices
     * define the subset of the training set for which the split is to be found. The initialStats are the sufficient
     * statistics before the data is split.
     */
    private SplitSpecification findBestSplitPointXg(int[] indices, Attribute attribute,
                                                  SufficientStatisticsXg initialStats) {
        double gSumLeft = 0; double gSumRight = 0; double hSumLeft = 0; double hSumRight = 0; double gain = 0;
        var statsRight = new SufficientStatisticsXg(initialStats.n, initialStats.sum, initialStats.sumOfGrad, initialStats.sumOfHess);
        var statsLeft = new SufficientStatisticsXg(0, 0.0, 0.0, 0.0);
        var splitSpecification = new SplitSpecification(attribute, 0.0, Double.NEGATIVE_INFINITY);
        double prevValue = Double.NEGATIVE_INFINITY;
        var xx = Arrays.stream(indices).mapToDouble(x -> data.instance(x).value(attribute)).toArray();
        var xxx = Utils.sortWithNoMissingValues(Arrays.stream(indices).mapToDouble(x ->
                data.instance(x).value(attribute)).toArray());
        var xxxx = Arrays.stream(Utils.sortWithNoMissingValues(Arrays.stream(indices).mapToDouble(x -> data.instance(x).value(attribute)).toArray())).map(x -> indices[x]).toArray();

        for(int i : Arrays.stream(Utils.sortWithNoMissingValues(Arrays.stream(indices).mapToDouble(x ->
                data.instance(x).value(attribute)).toArray())).map(x -> indices[x]).toArray()){
            Instance instance = data.instance(i);
            if(instance.value(attribute) > prevValue) {
                var splitQuality = splitQualityXg(initialStats, statsLeft, statsRight);
                if(splitQuality > splitSpecification.splitQuality){
                    splitSpecification.splitQuality = splitQuality;
                    splitSpecification.splitPoint = (instance.value(attribute) + prevValue) / 2.0;
                }
                prevValue = instance.value(attribute);
            }
            statsRight.updateStats(instance.classValue(), instance.weight(), false);
            statsLeft.updateStats(instance.classValue(), instance.weight(), true);
        }
        return  splitSpecification;
    }

    /**
     * Recursively grows a tree for a given set of data.
     */
    private int currDepth;
    private Node makeTree(int[] indices) {
        var stats = new SufficientStatisticsXg(0, 0.0, 0.0, 0.0);
        for (int i : indices) {
            stats.updateStats(data.instance(i).classValue(),data.instance(i).weight(), true);
        }
        if (stats.n <= 1 || this.currDepth >= max_depth) {
            return new LeafNode(this.calcWeight(stats.sumOfGrad, stats.sumOfHess));
        }
        var bestSplitSpecification = new SplitSpecification(null, Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY);
        for (Attribute attribute : selectRanAttributes(data, colsample_bynode)) {
            var SplitSpecification = findBestSplitPointXg(indices, attribute, stats);
            if (SplitSpecification.splitQuality > bestSplitSpecification.splitQuality) {
                bestSplitSpecification = SplitSpecification;
            }
        }
        if (bestSplitSpecification.splitQuality < gamma) {
            return new LeafNode(this.calcWeight(stats.sumOfGrad, stats.sumOfHess));
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

            // check min_child_weight
            double leftWeight = calculateSubsetWeight(leftSubset);
            double rightWeight = calculateSubsetWeight(rightSubset);
            if (leftWeight < min_child_weight || rightWeight < min_child_weight) {
                return new LeafNode(this.calcWeight(stats.sumOfGrad, stats.sumOfHess));
            }
            this.currDepth += 1;
            return new InternalNode(bestSplitSpecification.attribute, bestSplitSpecification.splitPoint,
                    makeTree(leftSubset.stream().mapToInt(Integer::intValue).toArray()),
                    makeTree(rightSubset.stream().mapToInt(Integer::intValue).toArray()));
        }
    }

    private List<Attribute> selectRanAttributes(Instances data, double colsample_bynode){
        List<Attribute> allAttributes = Collections.list(data.enumerateAttributes());
        int selectCount = (int)(allAttributes.size() * colsample_bynode);
        Random random = new Random(getSeed());
        Collections.shuffle(allAttributes, random);
        return allAttributes.subList(0, selectCount);
    }

    private double calculateSubsetWeight(ArrayList<Integer> subset) {
        double weightSum = 0.0;
        for (int i : subset) {
            weightSum += data.instance(i).weight();
        }
        return weightSum;
    }

    /** Returns the capabilities of the classifier: numeric predictors and numeric target. */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.BINARY_CLASS);
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
        this.currDepth = 0;
        // subsample indices
        int sampleSize = (int)(this.data.numInstances() * subsample);
        Random ran = new Random(getSeed());
        ArrayList<Integer> subIndices = new ArrayList<>();
        while(subIndices.size() < sampleSize) {
            int ranIndex = ran.nextInt(this.data.numInstances());
            if(!subIndices.contains(ranIndex)){
                subIndices.add(ranIndex);
            }
        }
        int[] subIndicesArr = subIndices.stream().mapToInt(Integer::intValue).toArray();
        rootNode = makeTree(subIndicesArr);
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