package neuronet;

import java.util.Random;

import static java.lang.Math.exp;

public class Main {
    public static void main(String[] args) {
        int hideNeuronsAmount = 1;
        int inputsAmount = 4;
        int epochs = 20_000;

        double[][] trainingInputs = new double[][] {
                {0, 0, 1, 1},
                {1, 1, 1, 1},
                {1, 0, 1, 1},
                {0, 1, 1, 1}};

        double[][] trainingOutputs = new double[][] {
                {0},
                {1},
                {1},
                {0}};

        double[][] weights = getRandomWeights(hideNeuronsAmount, inputsAmount);

        System.out.println("\nRandom weights:");
        printMatrix(transpose(weights));

        double[][] neuronsState = solveNeuronsState(trainingInputs, weights);
        double[][] neuronsOutput = sigmoid(neuronsState);

        System.out.println("\nOutputs of neurons:");
        printMatrix(neuronsOutput);

        for (int i = 0; i < epochs; i++) {
            neuronsState = solveNeuronsState(trainingInputs, weights);
            neuronsOutput = sigmoid(neuronsState);
            double[][] err = getError(neuronsOutput, trainingOutputs);
            double[][] adjustments = getAdjustments(trainingInputs, err, neuronsOutput);
            weights = correctWeights(weights, adjustments);
        }

        System.out.println("\nWeights after training:");
        printMatrix(transpose(weights));
        System.out.println("\nOutput after training:");
        printMatrix(neuronsOutput);

        // Test
        double[][] input = new double[][] {{0, 0, 0, 1}};
        neuronsState = solveNeuronsState(input, weights);
        neuronsOutput = sigmoid(neuronsState);

        System.out.println("\nNew situation:");
        printMatrix(neuronsOutput);
    }

    private static double[][] correctWeights(double[][] weights, double[][] adjustments) {
        int rows = weights.length;
        int cols = weights[0].length;
        double[][] newWeights = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                newWeights[i][j] = weights[i][j] + adjustments[i][j];
            }
        }
        return newWeights;
    }

    private static double[][] getAdjustments(double[][] input, double[][] err, double[][] output) {
        int inputSets = input.length;
        int neurons = err[0].length;
        double[][] temp = new double[inputSets][neurons];
        for (int i = 0; i < inputSets; i++) {
            for (int j = 0; j < neurons; j++) {
                temp[i][j] = err[i][j] * derivativeSigmoid(output[i][j]);
            }
        }
        return productMatrix(transpose(temp), input);
    }

    private static double[][] productMatrix(double[][] first, double[][] second) {
        int productRows = first.length;
        int productCols = second[0].length;
        int count = second.length;
        double[][] product = new double[productRows][productCols];
        for (int i = 0; i < productRows; i++) {
            for (int j = 0; j < productCols; j++) {
                product[i][j] = 0;
                for (int k = 0; k < count; k++) {
                    product[i][j] += first[i][k] * second[k][j];
                }
            }
        }
        return product;
    }

    private static double[][] getError(double[][] neuronsOutput, double[][] trainingOutputs) {
        int inputSets = neuronsOutput.length;
        int neurons = neuronsOutput[0].length;
        double[][] err = new double[inputSets][neurons];
        for (int i = 0; i < inputSets; i++) {
            for (int j = 0; j < neurons; j++) {
                err[i][j] = trainingOutputs[i][j] - neuronsOutput[i][j];
            }
        }
        return err;
    }

    private static double[][] solveNeuronsState(double[][] inputs, double[][] weights) {
        int neurons = weights.length;
        int inputSetsCount = inputs.length;
        double[][] neuronsState = new double[inputSetsCount][neurons];
        for (int i = 0; i < inputSetsCount; i++) {
            for (int j = 0; j < neurons; j++) {
                neuronsState[i][j] = getNeuronState(inputs[i], weights[j]);
            }
        }
        return neuronsState;
    }

    private static double getNeuronState(double[] inputs, double[] weights) {
        int count = inputs.length;
        double scalar = 0d;
        for (int i = 0; i < count; i++) {
            scalar += inputs[i] * weights[i];
        }
        return scalar;
    }

    private static void printMatrix(double[][] m) {
        int rows = m.length;
        int cols = m[0].length;

        System.out.print("[");
        for (int i = 0; i < rows; i++) {
            if (i != 0)
                System.out.print(" ");
            System.out.print("[");
            for (int j = 0; j < cols; j++) {
                System.out.print(m[i][j]);
                if (j != cols - 1)
                    System.out.print(", ");
            }
            System.out.print("]");
            if (i != rows - 1)
                System.out.println(",");
            else
                System.out.println("]");
        }
    }

    private static double[][] transpose(double[][] matrix) {
        double[][] trans = new double[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix[0].length; i++) {
            for (int j = 0; j < matrix.length; j++) {
                trans[i][j] = matrix[j][i];
            }
        }
        return trans;
    }

    private static double[][] getRandomWeights(int neurons, int inputs) {
        double[][] weights = new double[neurons][inputs];
        Random random = new Random(1);
        for (int i = 0; i < neurons; i++) {
            for (int j = 0; j < inputs; j++) {
                weights[i][j] = random.nextDouble() * 2 - 1;
            }
        }
        return weights;
    }

    public static double[][] sigmoid(double[][] x) {
        int rows = x.length;
        int cols = x[0].length;
        double[][] y = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                y[i][j] = sigmoid(x[i][j]);
            }
        }
        return y;
    }

    public static double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }

    public static double derivativeSigmoid(double sigmoid) {
        return sigmoid * (1 - sigmoid);
    }
}
