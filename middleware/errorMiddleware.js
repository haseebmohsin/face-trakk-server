function errorHandler(error, req, res, next) {
  // Default error status code
  let statusCode = res.statusCode || 500;

  // Check if the error is a CastError
  if (error.name === 'CastError') {
    statusCode = 400;
    error.message = 'Invalid ID format';
  }

  // Check if validation Error
  if (error.name === 'ValidationError') {
    statusCode = 400;
    error.message = error._message;
  }

  // Set the response status code
  res.status(statusCode);

  res.json({
    error: {
      message: error.message,
      ...(process.env.NODE_ENV === 'development' && { stack: error.stack }),
    },
  });
}

module.exports = errorHandler;
