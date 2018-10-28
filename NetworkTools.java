package uk.feconiz.nn;

/**
 * Provides tools needed for the usage of {@link Network}.
 * @author Panagiotis Karapas
 * @version 1.2
 */
@SuppressWarnings({"unused", "WeakerAccess"})
public class NetworkTools {

    /**
     * Creates a new array and initializes all of its cells with random numbers between the lower and upper bound (both inclusive).
     * @param size the size of the array to be created.
     * @param lower_bound the lower bound for the random values (inclusive).
     * @param upper_bound the upper bound for the random values (inclusive).
     * @return the new array.
     * @throws IllegalArgumentException if the lower bound is bigger than the upper bound.
     * @throws IllegalArgumentException if the size specified is &lt; 1.
     */
    public static double[] createRandomArray(int size, double lower_bound, double upper_bound) throws IllegalArgumentException{
        if(size < 0) throw new IllegalArgumentException("Array size can't be less than 0!");
        if(lower_bound > upper_bound) throw new IllegalArgumentException("Lower bound can't be bigger than the upper_bound!");

        double[] ar = new double[size];
        for(int i = 0; i < size; i++){
            ar[i] = randomValue(lower_bound,upper_bound);
        }
        return ar;
    }
    /**
     * Creates a new 2 dimensional array and initializes all of its cells with random numbers between the lower and upper bound (both inclusive).
     * @param sizeX the size of the arrays first dimension.
     * @param sizeY the size of the arrays second dimension.
     * @param lower_bound the lower bound for the random values (inclusive).
     * @param upper_bound the upper bound for the random values (inclusive).
     * @return the new array.
     * @throws IllegalArgumentException if the lower bound is bigger than the upper bound.
     * @throws IllegalArgumentException if either of the sizes specified is &lt; 1.
     */
    public static double[][] createRandomArray(int sizeX, int sizeY, double lower_bound, double upper_bound){
        if(sizeX < 0 || sizeY < 0)throw new IllegalArgumentException("Array size can't be less than 0!");
        double[][] ar = new double[sizeX][sizeY];
        for(int i = 0; i < sizeX; i++){
            ar[i] = createRandomArray(sizeY, lower_bound, upper_bound);
        }
        return ar;
    }

    /**
     * Returns a random double value between the lower and upper bound specified (both inclusive).
     * @param lower_bound the lower bound for the value.
     * @param upper_bound the upper bound for the value.
     * @return a random double value between the lower and upper bound specified (both inclusive)
     */
    public static double randomValue(double lower_bound, double upper_bound){
        if(lower_bound>upper_bound)throw new IllegalArgumentException("The lower bound must be smaller than the upper bound!");
        return Math.random()*(upper_bound-lower_bound) + lower_bound;
    }

}
