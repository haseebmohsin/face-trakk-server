const express = require('express');
const videoUploadRouter = require('./videoUploadRouter');

const router = express.Router();

// Mount routes
router.use('/video', videoUploadRouter);

module.exports = router;
