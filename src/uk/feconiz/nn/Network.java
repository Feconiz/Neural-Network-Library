package uk.feconiz.nn;

import uk.feconiz.exceptions.NotAFileException;
import uk.feconiz.exceptions.SizeMisMatchException;
import uk.feconiz.exceptions.TrainDataNotInitializedException;

import java.io.*;
import java.util.*;

/**
 * The network class provides a simple way of creating a Neural Network.
 * @author Panagiotis Karapas
 * @version 1.1
 */
@SuppressWarnings({"unused", "WeakerAccess"})
public class Network implements Serializable {
    private static final long serialVersionUID = 4026261013360883627L;
    private transient double[][] trainInput, trainOutput, testInput, testOutput;
    private PrintStream printStream;
    private boolean outputEnabled = false;
    private long timer = 0L;
    private int frequencyOfTrainingOutput = 0;
    private int frequencyOfEffectivenessOutput = 0;


    /**
     * The input size of this neural network. (The amount of input nodes)
     */
    public final int INPUT_SIZE;
    /**
     * The output size of this neural network. (The amount of output nodes)
     */
    public final int OUTPUT_SIZE;
    /**
     * The amount of layers (including the input and output layer) present in this neural network.
     */
    public final int NETWORK_SIZE;
    /**
     * The size of each layer (including the input and output layer) present in this neural network.
     */
    public final int[] NETWORK_LAYER_SIZES;

    private double[][] output;
    private double weight[][][];//Dimensions: Layer, neuron, connected neuron
    private double[][] bias;

    private double[][] error;//the current error from its neuron
    private double[][] outputDerivative;//output through the derivative of the sigmoid function

    /**
     * Creates and initializes a new neural network using the dimensions provided.
     * In more detail, this constructor will initialize all the arrays in this class and then set the values for bias and weight to random values (where -1 &lt;= value &lt;= 1.
     * @param NETWORK_LAYER_SIZES the sizes of each layer in the network (includes input and output layers).
     * @throws IllegalArgumentException if the size of the NETWORK_LAYER_SIZES is &lt; 2 or any of the layers has size &lt; 0.
     * @throws NullPointerException if the NETWORK_LAYER_SIZES argument is null.
     */
    public Network(int... NETWORK_LAYER_SIZES) throws IllegalArgumentException{
        if(NETWORK_LAYER_SIZES == null) throw new NullPointerException("NETWORK_LAYER_SIZES has to be initialized.");
        if(NETWORK_LAYER_SIZES.length < 2)throw new IllegalArgumentException("Layer size must be >= 2!");
        if(Arrays.stream(NETWORK_LAYER_SIZES).anyMatch(x->x<=0))throw new IllegalArgumentException("Every layer must have a size of at least 1 neuron!");

        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];
        this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
        this.OUTPUT_SIZE = NETWORK_LAYER_SIZES[NETWORK_SIZE - 1];


        this.output = new double[NETWORK_SIZE][];
        this.error = new double[NETWORK_SIZE][];
        this.outputDerivative = new double[NETWORK_SIZE][];
        this.weight = new double[NETWORK_SIZE][][];
        this.bias = new double[NETWORK_SIZE][];
        if(outputEnabled) printStream.println("Initializing layers with random values.");
        for (int i = 0; i < NETWORK_SIZE; i++) {
            this.output[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.error[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.outputDerivative[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.bias[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], -1, 1);
            if (i > 0) {
                weight[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], NETWORK_LAYER_SIZES[i - 1], -1, 1);
            }
        }
        if(outputEnabled) printStream.println("Initialization complete.");
    }
    /**
     * Enables the output for this NN, all output will be redirected to out.
     * @param out the output stream to redirect all output to.
     * @throws NullPointerException if out is null.
     */
    public void enableOutput(PrintStream out) throws NullPointerException {
        enableOutput(out, 0, 0);
    }
    /**
     * Enables the output for this NN, all output will be redirected to out.
     * @param out the output stream to redirect all output to.
     * @param frequencyOfTrainingOutput the amount of generations that pass before outputting anything to the output stream.
     * @throws NullPointerException if out is null.
     */
    public void enableOutput(PrintStream out, int frequencyOfTrainingOutput) throws NullPointerException {
        enableOutput(out, frequencyOfTrainingOutput, 0);
    }
    /**
     * Enables the output for this NN, all output will be redirected to out.
     * @param out the output stream to redirect all output to.
     * @param frequencyOfEffectivenessOutput the amount of generations that pass before calculating the score of the network and writing to the output stream.
     * @throws NullPointerException if out is null.
     */
    public void enableOutput(PrintStream out, Integer frequencyOfEffectivenessOutput) throws NullPointerException {
        enableOutput(out, 0, frequencyOfEffectivenessOutput);
    }
    /**
     * Enables the output for this NN, all output will be redirected to out.
     * @param out the output stream to redirect all output to.
     * @param frequencyOfTrainingOutput the amount of generations that pass before calculating the score of the network and writing to the output stream.
     * @param frequencyOfEffectivenessOutput the amount of generations that pass before outputting anything to the output stream.
     * @throws NullPointerException if out is null.
     */
    public void enableOutput(PrintStream out, int frequencyOfTrainingOutput, int frequencyOfEffectivenessOutput) throws NullPointerException{
        if(out == null)throw new NullPointerException("PrintStream assigned can't be null!");
        if(frequencyOfEffectivenessOutput < 0)throw new IllegalArgumentException("The frequency of effectiveness output can't be less than 0!");
        if(frequencyOfTrainingOutput < 0)throw new IllegalArgumentException("The frequency of training output can't be less than 0!");

        outputEnabled = true;
        printStream = out;
        printStream.println("Output Enabled");
    }
    /**
     * Trains the neural network for the amount of generations specified using the learning rate specified.
     * @param learningRate the rate the algorithm should use to learn (the bigger the fastest the algorithm will learn but the less precise it will be).
     * @param generations the amount of generations the training process should continue for.
     * @throws TrainDataNotInitializedException if the neural network has not had any data provided to it.
     * @throws IllegalArgumentException if the learning rate or generations number is &lt;= 0.
     */
    public void train(double learningRate, int generations) throws TrainDataNotInitializedException, IllegalArgumentException {
        if (trainInput == null || trainOutput == null)
            throw new TrainDataNotInitializedException("Can't train network if the data has not being initialized!");
        if(learningRate<=0) throw new IllegalArgumentException("Learning rate must be > 0!");
        if(generations<=0) throw new IllegalArgumentException("Generations number must be > 0!");

        try {
            train(trainInput, trainOutput, learningRate, generations);
        }catch (SizeMisMatchException e){
            e.printStackTrace();
        }
    }
    /**
     * Trains the neural network on the train data provided using the learning rate specified.
     * @param input the input data to use for training the network.
     * @param target the target output data to train the network towards.
     * @param learningRate the rate the algorithm should use to learn (the bigger the fastest the algorithm will learn but the less precise it will be, a good default value is 0.3).
     * @throws SizeMisMatchException if the size of the input or output arrays doesn't match the one expected.
     * @throws IllegalArgumentException if the learning rate is &lt;= 0.
     */
    public void train(double[] input, double[] target, double learningRate) throws SizeMisMatchException {
        if(input.length != INPUT_SIZE) throw new SizeMisMatchException("Given input doesn't match the networks input size!");
        if(target.length != OUTPUT_SIZE) throw new SizeMisMatchException("Given output doesn't match the networks output size!");
        if(learningRate<=0) throw new IllegalArgumentException("Learning rate must be > 0!");

        calculate(input);
        calcError(target);
        updateWeightsAndBiases(learningRate);
    }
    /**
     * Trains the neural network on the train data provided for the amount of generations specified using the learning rate specified.
     * @param input the input data to use for training the network.
     * @param target the target output data to train the network towards.
     * @param learningRate the rate the algorithm should use to learn (the bigger the fastest the algorithm will learn but the less precise it will be, a good default value is 0.3).
     * @param generations the amount of generations the training process should continue for.
     * @throws SizeMisMatchException if the size of the input or output arrays doesn't match the one expected.
     * @throws IllegalArgumentException if the learning rate or generations number is &lt;= 0.
     */
    public void train(double[][] input, double[][] target, double learningRate, int generations) throws SizeMisMatchException {
        if(Arrays.stream(input).anyMatch(x->x.length != INPUT_SIZE)) throw new SizeMisMatchException("Given input doesn't match the networks input size!");
        if(Arrays.stream(target).anyMatch(x->x.length != OUTPUT_SIZE)) throw new SizeMisMatchException("Given output doesn't match the networks output size!");
        if(learningRate<=0) throw new IllegalArgumentException("Learning rate must be > 0!");
        if(generations<=0) throw new IllegalArgumentException("Generations number must be > 0!");

        for (int i = 0; i < generations; i++) {
            if(frequencyOfTrainingOutput > 0 && generations%frequencyOfTrainingOutput == 0){
                printStream.println("Starting generation number " + i + ".");
            }
            if(frequencyOfEffectivenessOutput > 0 && generations%frequencyOfEffectivenessOutput == 0){
                printStream.println("The verage deviation is " + getAverageDeviation() + "(smaller is better) ");
            }
            for (int j = 0; j < input.length; j++) {
                train(input[j], target[j], learningRate);
            }
        }

    }
    /**
     * Updates the weights and biases of all the neurons using the specified learning rate.
     * @param learningRate the learning rate to use when updating the biases and weights.
     * @throws IllegalArgumentException if the learning rate or generations number is &lt;= 0.
     */
    private void updateWeightsAndBiases(double learningRate) throws IllegalArgumentException{
        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                double delta = -learningRate * error[layer][neuron];//calculate the delta we need to change the bias by
                bias[layer][neuron] += delta;

                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {
                    weight[layer][neuron][prevNeuron] += delta * output[layer - 1][prevNeuron];
                }
            }
        }
    }
    /**
     * Calculates the error between the output and the target output.
     * @param target the target output.
     */
    private void calcError(double[] target) {
        for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[NETWORK_SIZE - 1]; neuron++) {
            error[NETWORK_SIZE - 1][neuron] = (output[NETWORK_SIZE - 1][neuron] - target[neuron]) * outputDerivative[NETWORK_SIZE - 1][neuron];
        }
        for (int layer = NETWORK_SIZE - 2; layer > 0; layer--) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                double sum = 0;
                for (int nextNeuron = 0; nextNeuron < NETWORK_LAYER_SIZES[layer + 1]; nextNeuron++) {
                    sum += weight[layer + 1][nextNeuron][neuron] * error[layer + 1][nextNeuron];
                }
                error[layer][neuron] = sum * outputDerivative[layer][neuron];
            }
        }
    }
    /**
     * Calculate and return the output given the specified input.
     * @param input the input to use for the calculations.
     * @return the output after doing all the calculations.
     * @throws SizeMisMatchException if the size of the input array doesn't match the one expected.
     */
    public double[] calculate(double... input) throws SizeMisMatchException {
        if(input.length != INPUT_SIZE) throw new SizeMisMatchException("Given input doesn't match the networks input size!");


        this.output[0] = input;

        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++) {
                double sum = bias[layer][neuron];
                for (int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer - 1]; prevNeuron++) {

                    sum += output[layer - 1][prevNeuron] * weight[layer][neuron][prevNeuron];
                }
                output[layer][neuron] = sigmoid(sum);
                outputDerivative[layer][neuron] = output[layer][neuron] * (1 - output[layer][neuron]);
            }
        }
        return output[NETWORK_SIZE - 1];
    }
    /**
     * Calculates and returns the value of 1/(1+(e^-x))
     * @param x the value to apply the function on.
     * @return the result of the function.
     */
    private double sigmoid(double x) {
        return 1d / (1 + Math.exp(-x));
    }
    /**
     * Reads the file from the specified location and splits it into train and test data using the specified percentage.
     * @param filepath the filepath to the file used as input.
     * @param percentTrain the percentage of the inputs/outputs to be used as training data (range: (0-1]).
     * @throws FileNotFoundException if the specified filepath doesn't exist.
     * @throws NotAFileException if the specified filepath doesn't point to a file.
     * @throws IllegalArgumentException if the percentage specified doesn't fall into the rage (0-1].
     */
    public void readAndSplitData(String filepath, double percentTrain) throws FileNotFoundException, NotAFileException, IllegalArgumentException {
        File f = new File(filepath);
        if(!f.isFile())throw new NotAFileException("The specified filepath doesn't point to a file!");
        if(!f.exists())throw new FileNotFoundException("The specified filepath doesn't exist!");
        if(percentTrain <= 0 || percentTrain > 1) throw new IllegalArgumentException("The percent of train data must be in the range (0-1]!");

        if(outputEnabled) printStream.println("Reading file.");
        ArrayList<double[][]> result = new ArrayList<>();
        try (Scanner sc = new Scanner(f)) {
            while (sc.hasNextLine()) {
                double[][] current = new double[2][];
                current[0] = new double[INPUT_SIZE];
                current[1] = new double[OUTPUT_SIZE];

                for (int i = 0; i < INPUT_SIZE; i++) {
                    current[0][i] = sc.nextDouble();
                }
                for (int i = 0; i < OUTPUT_SIZE; i++) {
                    current[1][i] = sc.nextDouble();
                }
                if (sc.hasNextLine()) sc.nextLine();
                result.add(current);
            }
        }
        splitAndSetData(percentTrain, result);
    }
    /**
     * Splits the data to train and test data using the percentage specified.
     * @param percentTrain the percent of the data to be used for the training process.
     * @param result the data to split.
     */
    public void splitAndSetData(double percentTrain, ArrayList<double[][]> result) {
        if(outputEnabled) printStream.println("Splitting Data");

        Collections.shuffle(result);
        double[][][] data = result.toArray(new double[0][][]);
        double[][] input = Arrays.stream(data).map(x -> x[0]).toArray(double[][]::new);
        double[][] output = Arrays.stream(data).map(x -> x[1]).toArray(double[][]::new);

        trainInput = new double[(int) Math.round(input.length * percentTrain)][INPUT_SIZE];
        trainOutput = new double[(int) Math.round(output.length * percentTrain)][OUTPUT_SIZE];
        testInput = new double[input.length - trainInput.length][INPUT_SIZE];
        testOutput = new double[output.length - trainOutput.length][OUTPUT_SIZE];

        for (int i = 0; i < trainInput.length; i++) {
            trainInput[i] = input[i];
            trainOutput[i] = output[i];
        }
        for (int i = input.length - 1; i > trainInput.length; i--) {
            testInput[i - trainInput.length-1] = input[i];
            testOutput[i - trainInput.length-1] = output[i];
        }
        if(outputEnabled) printStream.println("Done splitting data!");

    }
    /**
     * Calculates all the test inputs and averages the difference between the actual and expected outputs.
     * @return the average deviation of expected and actual outputs.
     * @throws SizeMisMatchException if the test input or output has a size of 0.
     * @throws NullPointerException if the test input or output is null.
     * @throws SizeMisMatchException if the size of the input or output doesn't match the size of the networks inputs and outputs.
     */
    public double getAverageDeviation() throws  NullPointerException, SizeMisMatchException {
        if(testInput == null)throw new NullPointerException("Test input has not being initialized!");
        if(testOutput == null)throw new NullPointerException("Test output has not being initialized!");
        if(testInput.length == 0) throw new SizeMisMatchException("Test input has length 0!");
        if(testOutput.length == 0) throw new SizeMisMatchException("Test output has length 0!");


        double sum = 0;
        for(int i = 0; i < testInput.length;i++){
            double[] result = calculate(testInput[i]);
            for(int j = 0; j < OUTPUT_SIZE; j++) {
                sum += Math.abs(result[j] - testOutput[i][j]);
            }
        }
        return sum/(testInput.length* OUTPUT_SIZE);
    }
    /**
     * Saves this neural network to the file specified.
     * @param filepath the filepath to the file to use for saving.
     * @throws FileNotFoundException if the specified filepath doesn't exist.
     * @throws NotAFileException if the specified filepath doesn't point to a file.
     */
    public void save(String filepath) throws NotAFileException, FileNotFoundException {
        if(outputEnabled){
            printStream.println("Starting the saving process.");
            timer = System.currentTimeMillis();
        }
        File f = new File(filepath);
        if(!f.isFile())throw new NotAFileException("The specified filepath doesn't point to a file!");
        if(!f.exists())throw new FileNotFoundException("The specified filepath doesn't exist!");
        if(outputEnabled){
            printStream.println("Starting the saving process.");
            timer = System.currentTimeMillis();
        }

        try {
            FileOutputStream fileOut = new FileOutputStream(f);
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(this);
            out.close();
            fileOut.close();
        } catch (IOException i) {
            i.printStackTrace();
        }
        if(outputEnabled) printStream.println("Saving finished after " + (System.currentTimeMillis() - timer) + "ms(" + (System.currentTimeMillis() - timer)/1000 + "s).");
    }
    /**
     * Loads a neural network from the file specified.
     * @return the neural network loaded.
     * @param filepath the filepath to the file to use for loading.
     * @throws FileNotFoundException if the specified filepath doesn't exist.
     * @throws NotAFileException if the specified filepath doesn't point to a file.
     */
    public Network load(String filepath) throws IOException {
        if(outputEnabled){
            printStream.println("Starting the loading process.");
            timer = System.currentTimeMillis();
        }
        File f = new File(filepath);
        if(!f.isFile())throw new NotAFileException("The specified filepath doesn't point to a file!");
        if(!f.exists())throw new FileNotFoundException("The specified filepath doesn't exist!");

        try {
            FileInputStream fileIn = new FileInputStream(f);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            Network net = (Network) in.readObject();
            in.close();
            fileIn.close();
            if(outputEnabled) printStream.println("Loading finished after " + (System.currentTimeMillis() - timer) + "ms(" + (System.currentTimeMillis() - timer)/1000 + "s).");
            return net;
        } catch (ClassNotFoundException c) {
            c.printStackTrace();
        }
        return null;
    }
    /**
     * Returns the provided training input.
     * @return the provided training input.
     */
    public double[][] getTrainInput() {
        return trainInput;
    }
    /**
     * Checks that the training inputs provided are intact and if so sets them as the training data.
     * @param trainInput the training inputs to be set.
     * @throws IllegalArgumentException if the size of any of the training inputs specified doesn't match the neural networks input size.
     * @throws NullPointerException if the training inputs specified are null.
     */
    public void setTrainInput(double[][] trainInput) throws NullPointerException, IllegalArgumentException{
        if(trainInput == null) throw new NullPointerException("The training input can't be set to null!");
        if(Arrays.stream(trainInput).anyMatch(x->x.length != INPUT_SIZE)) throw new IllegalArgumentException("All training input sizes must match the neural networks input size!");
        this.trainInput = trainInput;
    }
    /**
     * Returns the provided expected outputs for training.
     * @return the provided expected outputs for training.
     */
    public double[][] getTrainOutput() {
        return trainOutput;
    }
    /**
     * Checks that the training outputs provided are intact and if so sets them as the training data.
     * @param trainOutput the training outputs to be set.
     * @throws IllegalArgumentException if the size of any of the training outputs specified doesn't match the neural networks output size.
     * @throws NullPointerException if the training outputs specified are null.
     */
    public void setTrainOutput(double[][] trainOutput) {
        if(trainOutput == null) throw new NullPointerException("The training output can't be set to null!");
        if(Arrays.stream(trainOutput).anyMatch(x->x.length != OUTPUT_SIZE)) throw new IllegalArgumentException("All training output sizes must match the neural networks output size!");
        this.trainOutput = trainOutput;
    }
    /**
     * Returns the provided testing input.
     * @return the provided testing input.
     */
    public double[][] getTestInput() {
        return testInput;
    }
    /**
     * Checks that the testing inputs provided are intact and if so sets them as the testing data.
     * @param testInput the testing inputs to be set.
     * @throws IllegalArgumentException if the size of any of the testing inputs specified doesn't match the neural networks input size.
     * @throws NullPointerException if the testing inputs specified are null.
     */
    public void setTestInput(double[][] testInput) {
        if(testInput == null) throw new NullPointerException("The testing input can't be set to null!");
        if(Arrays.stream(testInput).anyMatch(x->x.length != INPUT_SIZE)) throw new IllegalArgumentException("All testing input sizes must match the neural networks input size!");
        this.testInput = testInput;
    }
    /**
     * Returns the provided expected outputs for testing.
     * @return the provided expected outputs for testing.
     */
    public double[][] getTestOutput() {
        return testOutput;
    }
    /**
     * Checks that the testing outputs provided are intact and if so sets them as the training data.
     * @param testOutput the testing outputs to be set.
     * @throws IllegalArgumentException if the size of any of the testing outputs specified doesn't match the neural networks output size.
     * @throws NullPointerException if the testing outputs specified are null.
     */
    public void setTestOutput(double[][] testOutput) {
        if(testOutput == null) throw new NullPointerException("The testing output can't be set to null!");
        if(Arrays.stream(testOutput).anyMatch(x->x.length != OUTPUT_SIZE)) throw new IllegalArgumentException("All testing output sizes must match the neural networks output size!");
        this.testOutput = testOutput;
    }
    /**
     * Returns the output stream used in this neural network, null if the output stream isn't set or enabled.
     * @return the provided expected outputs for testing.
     */
    public PrintStream getPrintStream() {
        return outputEnabled?printStream:null;
    }
    /**
     * Changes the current output stream to the one specified.
     * @param printStream the new output stream to use.
     */
    public void changePrintStream(PrintStream printStream) {
        if(!outputEnabled) throw new IllegalStateException("Use enableOutput to set the output stream for the first time!");
        this.printStream = printStream;
    }
    /**
     * Returns the amount of generations that pass before outputting anything to the output stream.
     * @return the amount of generations that pass before outputting anything to the output stream.
     */
    public int getFrequencyOfTrainingOutput() {
        return frequencyOfTrainingOutput;
    }
    /**
     * Sets the amount of generations that pass before outputting anything to the output stream.
     * @param frequencyOfTrainingOutput the new amount of generations that pass before outputting anything to the output stream.
     */
    public void setFrequencyOfTrainingOutput(int frequencyOfTrainingOutput) {
        if(!outputEnabled) throw new IllegalStateException("Use enableOutput to set the output stream first!");
        this.frequencyOfTrainingOutput = frequencyOfTrainingOutput;
    }
    /**
     * Returns the amount of generations that pass before calculating the score of the network and writing to the output stream.
     * @return the amount of generations that pass before calculating the score of the network and writing to the output stream.
     * @see #getAverageDeviation()
     */
    public int getFrequencyOfEffectivenessOutput() {
        return frequencyOfEffectivenessOutput;
    }
    /**
     * Sets the amount of generations that pass before calculating the score of the network and writing to the output stream.
     * Note: the bigger the number gets it will affect training time more and more.
     * @param frequencyOfEffectivenessOutput the new frequency in which the training algorithm should print the average deviation (using the test data).
     * @see #getAverageDeviation()
     */
    public void setFrequencyOfEffectivenessOutput(int frequencyOfEffectivenessOutput) {
        if(!outputEnabled) throw new IllegalStateException("Use enableOutput to set the output stream first!");
        if(frequencyOfEffectivenessOutput < 1) throw new IllegalArgumentException("The frequency of the effectiveness output can't be less than 1!");
        this.frequencyOfEffectivenessOutput = frequencyOfEffectivenessOutput;
    }
}
