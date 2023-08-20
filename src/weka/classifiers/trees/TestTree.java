package weka.classifiers.trees;

import weka.classifiers.RandomizableClassifier;
import weka.core.*;

public class TestTree extends RandomizableClassifier implements WeightedInstancesHandler {

    protected Node root;

    protected class Node {
        double prediction;
        int attribute;
        double splitValue;
        Node left;
        Node right;

        public Node() {
            this.prediction = 0.0;
            this.attribute = -1;
            this.splitValue = Double.NaN;
        }
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        root = new Node();
        buildTree(root, data);
    }

    protected void buildTree(Node node, Instances data) throws Exception {
        if (data.numInstances() == 0) {
            return;
        }

        double totalGradient = 0.0;
        double totalHessian = 0.0;

        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            totalGradient += instance.classValue();
            totalHessian += instance.weight();
        }

        node.prediction = -totalGradient / totalHessian;

        double bestGain = 0.0;
        int bestAttribute = -1;
        double bestSplitValue = Double.NaN;
        Instances[] bestSplit = null;

        for (int attribute = 0; attribute < data.numAttributes(); attribute++) {
            if (attribute != data.classIndex()) {
                Instances[] split = splitData(attribute, data);
                double gain = calculateGain(split, totalGradient, totalHessian);
                if (gain > bestGain) {
                    bestGain = gain;
                    bestAttribute = attribute;
                    bestSplitValue = data.meanOrMode(attribute);
                    bestSplit = split;
                }
            }
        }

        if (bestGain > 0.0) {
            node.attribute = bestAttribute;
            node.splitValue = bestSplitValue;
            node.left = new Node();
            node.right = new Node();
            buildTree(node.left, bestSplit[0]);
            buildTree(node.right, bestSplit[1]);
        }
    }

    protected Instances[] splitData(int attribute, Instances data) throws Exception {
        Instances[] split = new Instances[2];
        split[0] = new Instances(data, data.numInstances());
        split[1] = new Instances(data, data.numInstances());
        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            if (instance.isMissing(attribute) || instance.value(attribute) <= data.meanOrMode(attribute)) {
                split[0].add(instance);
            } else {
                split[1].add(instance);
            }
        }
        return split;
    }

    protected double calculateGain(Instances[] split, double totalGradient, double totalHessian) {
        double leftGradient = 0.0;
        double rightGradient = 0.0;
        double leftHessian = 0.0;
        double rightHessian = 0.0;

        for (int i = 0; i < split[0].numInstances(); i++) {
            Instance instance = split[0].instance(i);
            leftGradient += instance.classValue(); // Gradient
            leftHessian += instance.weight();      // Hessian
        }

        for (int i = 0; i < split[1].numInstances(); i++) {
            Instance instance = split[1].instance(i);
            rightGradient += instance.classValue(); // Gradient
            rightHessian += instance.weight();      // Hessian
        }

        double lambda = 1.0; // 正则化参数，可根据需要进行调整

        double gain = 0.5 * (
                Math.pow(leftGradient, 2) / (leftHessian + lambda)
                        + Math.pow(rightGradient, 2) / (rightHessian + lambda)
                        - Math.pow(totalGradient, 2) / (totalHessian + lambda)
        );

        return gain;
    }


    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return classifyInstance(instance, root);
    }

    protected double classifyInstance(Instance instance, Node node) throws Exception {
        if (node.attribute == -1) {
            return node.prediction;
        } else {
            if (instance.isMissing(node.attribute) || instance.value(node.attribute) <= node.splitValue) {
                return classifyInstance(instance, node.left);
            } else {
                return classifyInstance(instance, node.right);
            }
        }
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        // 为分类器设置适当的能力，例如处理数值特征、分类特征等
        return result;
    }
}
