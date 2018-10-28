package uk.feconiz.exceptions;
/**
 * Signals that some data sizes did not match with the requirement.
 */
public class SizeMisMatchException extends Exception {
    /**
     * Constructs a SizeMisMatchException with the specified detail message.
     * @param message The detail message.
     */
    public SizeMisMatchException(String message) {
        super(message);
    }
}
