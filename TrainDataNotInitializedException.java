package uk.feconiz.exceptions;

/**
 * Signals that some required data was not provided.
 *
 */
@SuppressWarnings({"unused", "WeakerAccess"})
public class TrainDataNotInitializedException extends Exception{

    private static final long serialVersionUID = -5496381396519104016L;

    /**
     * Constructs a TrainDataNotInitializedException with the specified detail message.
     * @param message The detail message.
     */
    public TrainDataNotInitializedException(String message) {
        super(message);
    }
}
