const express = require('express');
const router = express.Router();
const videoUploadRouter = require('./videoUploadRouter');
const personRouter = require('./personRouter');
const facedashRouter = require('./facedashRouter');

// Mount routes for video upload
router.use('/video', videoUploadRouter);

// Mount routes for person
router.use('/person', personRouter);

// Mount routes for person
router.use('/facedash', facedashRouter);

module.exports = router;
