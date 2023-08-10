const express = require('express');
const router = express.Router();
const {
  uploadVideo,
  getClustersData,
  getClusterDataById,
  createAdditionalTrainingDatasets,
  startTraining,
  moveToNewCluster,
} = require('../controllers/trainingController');

// GET /api/getFaces
router.get('/getClustersData', getClustersData);

// GET /api/getClusterData/:id
router.get('/getClusterData/:id', getClusterDataById);

// POST /api/uploadVideo
router.post('/uploadVideo', uploadVideo);

// POST /api/moveToNewCluster
router.post('/moveToNewCluster', moveToNewCluster);

// POST /api/startTraining
router.post('/startTraining', startTraining);

// POST /api/createAdditionalTrainingDatasets
router.post('/createAdditionalTrainingDatasets', createAdditionalTrainingDatasets);

module.exports = router;
