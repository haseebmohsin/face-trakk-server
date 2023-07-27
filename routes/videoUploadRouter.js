const express = require('express');
const router = express.Router();
const {
  videoUpload,
  getClustersData,
  getClusterDataById,
  xmlUpdate,
  startTraining,
  moveToCluster,
} = require('../controllers/videoUploadController');

// GET /api/getFaces
router.get('/getClustersData', getClustersData);

// GET /api/getClusterData/:id
router.get('/getClusterData/:id', getClusterDataById);

// POST /api/videoUpload
router.post('/upload', videoUpload);

// POST /api/xmlUpdate
router.post('/xmlUpdate', xmlUpdate);

// POST /api/startTraining
router.post('/startTraining', startTraining);

// POST /api/moveToCluster
router.post('/moveToCluster', moveToCluster);

module.exports = router;
