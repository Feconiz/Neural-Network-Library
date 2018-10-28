package uk.feconiz.exceptions;
/**
 * Signals that an array or other data type had a size that was outside the acceptable range.
 */
public class WrongSizeException extends Exception {
    /**
     * Constructs a WrongSizeException with the specified detail message.
     * @param message The detail message.
     */
    public WrongSizeException(String message) {
        super(message);
    }
}
