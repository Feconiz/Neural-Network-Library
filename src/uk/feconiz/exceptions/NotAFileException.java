package uk.feconiz.exceptions;

import java.io.IOException;

/**
 * Signals that a filepath doesn't point to a file (possibly a directory).
 */
public class NotAFileException extends IOException {
    /**
     * Constructs a NotAFileException with the specified detail message.
     * @param message The detail message.
     */
    public NotAFileException(String message) {
        super(message);
    }
}
