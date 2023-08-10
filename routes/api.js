const express = require('express');
const router = express.Router();
const trainingRouter = require('./trainingRouter');
const personRouter = require('./personRouter');
const facedashRouter = require('./facedashRouter');

// Mount routes for video upload
router.use('/training', trainingRouter);

// Mount routes for person
router.use('/person', personRouter);

// Mount routes for person
router.use('/facedash', facedashRouter);

module.exports = router;
