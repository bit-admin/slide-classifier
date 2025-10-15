#!/usr/bin/env node
/**
 * Slide Classification Inference Script (Node.js)
 * Uses ONNX Runtime to classify slide images with the trained MobileNetV4 model
 */

const fs = require('fs').promises;
const path = require('path');
const { program } = require('commander');
const ort = require('onnxruntime-node');
const sharp = require('sharp');

class SlideClassifier {
    /**
     * Initialize the classifier
     * @param {string} modelPath - Path to the ONNX model file
     * @param {string} modelInfoPath - Path to the model info JSON file
     * @param {number} confidenceThreshold - Minimum confidence for predictions
     */
    constructor(modelPath, modelInfoPath, confidenceThreshold = 0.5) {
        this.modelPath = modelPath;
        this.modelInfoPath = modelInfoPath;
        this.confidenceThreshold = confidenceThreshold;
        this.session = null;
        this.modelInfo = null;
        this.classNames = null;
    }

    /**
     * Load the ONNX model and metadata
     */
    async loadModel() {
        try {
            // Check if model file exists
            await fs.access(this.modelPath);
            console.log(`Loading ONNX model from: ${this.modelPath}`);

            // Load ONNX model
            this.session = await ort.InferenceSession.create(this.modelPath);

            // Load model info
            if (this.modelInfoPath) {
                try {
                    await fs.access(this.modelInfoPath);
                    const modelInfoData = await fs.readFile(this.modelInfoPath, 'utf8');
                    this.modelInfo = JSON.parse(modelInfoData);
                    this.classNames = this.modelInfo.class_names;

                    console.log(`Found ${this.classNames.length} classes: ${this.classNames.join(', ')}`);

                    if (this.modelInfo.training_info) {
                        const { best_val_acc, epoch } = this.modelInfo.training_info;
                        console.log(`Model trained for ${epoch} epochs, best validation accuracy: ${best_val_acc}%`);
                    }
                } catch (error) {
                    console.warn(`Warning: Could not load model info from ${this.modelInfoPath}`);
                    console.warn('Using default class indices');
                    // Fallback: assume classes are indexed 0, 1, 2, etc.
                    this.classNames = Array.from({length: 7}, (_, i) => `class_${i}`);
                }
            }

            console.log('Model loaded successfully');
        } catch (error) {
            throw new Error(`Failed to load model: ${error.message}`);
        }
    }

    /**
     * Preprocess image for model input
     * @param {string} imagePath - Path to the image file
     * @returns {Promise<Float32Array>} Preprocessed image tensor
     */
    async preprocessImage(imagePath) {
        try {
            // Load and resize image to 256x256
            const imageBuffer = await sharp(imagePath)
                .resize(256, 256)
                .raw()
                .toBuffer();

            // Convert to RGB if needed and normalize
            const rgbBuffer = await sharp(imageBuffer, {
                raw: { width: 256, height: 256, channels: 3 }
            }).raw().toBuffer();

            // Convert to Float32Array and apply ImageNet normalization
            const float32Data = new Float32Array(3 * 256 * 256);
            const mean = [0.485, 0.456, 0.406];
            const std = [0.229, 0.224, 0.225];

            for (let c = 0; c < 3; c++) {
                for (let h = 0; h < 256; h++) {
                    for (let w = 0; w < 256; w++) {
                        const pixelIndex = h * 256 + w;
                        const rgbIndex = pixelIndex * 3 + c;
                        const tensorIndex = c * 256 * 256 + h * 256 + w;

                        // Normalize pixel value to [0, 1] then apply ImageNet normalization
                        const normalizedPixel = (rgbBuffer[rgbIndex] / 255.0 - mean[c]) / std[c];
                        float32Data[tensorIndex] = normalizedPixel;
                    }
                }
            }

            return float32Data;
        } catch (error) {
            throw new Error(`Failed to preprocess image ${imagePath}: ${error.message}`);
        }
    }

    /**
     * Apply softmax to logits
     * @param {Float32Array} logits - Raw model outputs
     * @returns {Float32Array} Softmax probabilities
     */
    softmax(logits) {
        const maxLogit = Math.max(...logits);
        const expLogits = logits.map(x => Math.exp(x - maxLogit));
        const sumExp = expLogits.reduce((sum, x) => sum + x, 0);
        return expLogits.map(x => x / sumExp);
    }

    /**
     * Predict class for a single image
     * @param {string} imagePath - Path to the image file
     * @param {boolean} returnProbabilities - Whether to return class probabilities
     * @returns {Promise<Object>} Prediction results
     */
    async predictSingle(imagePath, returnProbabilities = false) {
        try {
            // Preprocess image
            const inputTensor = await this.preprocessImage(imagePath);

            // Create input tensor for ONNX Runtime
            const tensor = new ort.Tensor('float32', inputTensor, [1, 3, 256, 256]);

            // Run inference
            const feeds = { input: tensor };
            const results = await this.session.run(feeds);

            // Get output tensor
            const outputTensor = results.output;
            const logits = Array.from(outputTensor.data);

            // Apply softmax to get probabilities
            const probabilities = this.softmax(logits);

            // Find predicted class
            const maxProbIndex = probabilities.indexOf(Math.max(...probabilities));
            const predictedClass = this.classNames[maxProbIndex];
            const confidence = probabilities[maxProbIndex];

            const result = {
                image_path: imagePath,
                predicted_class: predictedClass,
                confidence: confidence,
                above_threshold: confidence >= this.confidenceThreshold
            };

            if (returnProbabilities) {
                const classProbabilities = {};
                this.classNames.forEach((className, index) => {
                    classProbabilities[className] = probabilities[index];
                });
                result.class_probabilities = classProbabilities;
            }

            return result;

        } catch (error) {
            return {
                image_path: imagePath,
                error: error.message,
                predicted_class: null,
                confidence: 0.0,
                above_threshold: false
            };
        }
    }

    /**
     * Predict classes for multiple images
     * @param {string[]} imagePaths - Array of image file paths
     * @param {boolean} returnProbabilities - Whether to return class probabilities
     * @param {boolean} showProgress - Whether to show progress
     * @returns {Promise<Object[]>} Array of prediction results
     */
    async predictBatch(imagePaths, returnProbabilities = false, showProgress = true) {
        const results = [];
        const totalImages = imagePaths.length;

        for (let i = 0; i < imagePaths.length; i++) {
            const imagePath = imagePaths[i];

            if (showProgress && i % 10 === 0) {
                console.log(`Processing image ${i + 1}/${totalImages}: ${path.basename(imagePath)}`);
            }

            const result = await this.predictSingle(imagePath, returnProbabilities);
            results.push(result);
        }

        return results;
    }

    /**
     * Predict classes for all images in a directory
     * @param {string} directoryPath - Path to directory containing images
     * @param {boolean} returnProbabilities - Whether to return class probabilities
     * @param {boolean} showProgress - Whether to show progress
     * @returns {Promise<Object[]>} Array of prediction results
     */
    async predictDirectory(directoryPath, returnProbabilities = false, showProgress = true) {
        try {
            // Check if directory exists
            await fs.access(directoryPath);

            // Find all image files
            const imageExtensions = new Set(['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']);
            const files = await fs.readdir(directoryPath);

            const imagePaths = files
                .filter(file => imageExtensions.has(path.extname(file).toLowerCase()))
                .map(file => path.join(directoryPath, file))
                .sort();

            if (imagePaths.length === 0) {
                console.log(`No images found in ${directoryPath}`);
                return [];
            }

            console.log(`Found ${imagePaths.length} images in ${directoryPath}`);
            return await this.predictBatch(imagePaths, returnProbabilities, showProgress);

        } catch (error) {
            throw new Error(`Directory not found or inaccessible: ${directoryPath}`);
        }
    }
}

/**
 * Print a summary of prediction results
 * @param {Object[]} results - Array of prediction results
 */
function printResultsSummary(results) {
    if (!results || results.length === 0) {
        console.log('No results to summarize');
        return;
    }

    // Count predictions by class
    const classCounts = {};
    const confidenceScores = [];
    let errors = 0;
    let aboveThreshold = 0;

    results.forEach(result => {
        if (result.error) {
            errors++;
        } else {
            classCounts[result.predicted_class] = (classCounts[result.predicted_class] || 0) + 1;
            confidenceScores.push(result.confidence);
            if (result.above_threshold) {
                aboveThreshold++;
            }
        }
    });

    const totalProcessed = results.length;
    const successful = totalProcessed - errors;

    console.log('\n' + '='.repeat(60));
    console.log('PREDICTION SUMMARY');
    console.log('='.repeat(60));
    console.log(`Total images processed: ${totalProcessed}`);
    console.log(`Successful predictions: ${successful}`);
    console.log(`Errors: ${errors}`);
    console.log(`Above confidence threshold: ${aboveThreshold}/${successful}`);

    if (confidenceScores.length > 0) {
        const avgConfidence = confidenceScores.reduce((sum, score) => sum + score, 0) / confidenceScores.length;
        console.log(`Average confidence: ${avgConfidence.toFixed(3)}`);
    }

    console.log('\nClass Distribution:');
    console.log('-'.repeat(40));

    Object.entries(classCounts)
        .sort(([,a], [,b]) => b - a)
        .forEach(([className, count]) => {
            const percentage = successful > 0 ? (count / successful) * 100 : 0;
            console.log(`${className.padEnd(35)} ${count.toString().padStart(4)} (${percentage.toFixed(1).padStart(5)}%)`);
        });
}

/**
 * Print detailed prediction results
 * @param {Object[]} results - Array of prediction results
 * @param {boolean} showProbabilities - Whether to show class probabilities
 * @param {number|null} maxResults - Maximum number of results to display
 */
function printDetailedResults(results, showProbabilities = false, maxResults = null) {
    if (!results || results.length === 0) {
        console.log('No results to display');
        return;
    }

    console.log('\n' + '='.repeat(80));
    console.log('DETAILED RESULTS');
    console.log('='.repeat(80));

    const displayResults = maxResults ? results.slice(0, maxResults) : results;

    displayResults.forEach((result, index) => {
        const imageName = path.basename(result.image_path);

        if (result.error) {
            console.log(`${(index + 1).toString().padStart(3)}. ${imageName}`);
            console.log(`     ERROR: ${result.error}`);
        } else {
            const confidenceIndicator = result.above_threshold ? '✓' : '✗';
            console.log(`${(index + 1).toString().padStart(3)}. ${imageName}`);
            console.log(`     Class: ${result.predicted_class}`);
            console.log(`     Confidence: ${result.confidence.toFixed(3)} ${confidenceIndicator}`);

            if (showProbabilities && result.class_probabilities) {
                console.log('     All probabilities:');
                const sortedProbs = Object.entries(result.class_probabilities)
                    .sort(([,a], [,b]) => b - a);

                sortedProbs.forEach(([className, prob]) => {
                    console.log(`       ${className.padEnd(30)} ${prob.toFixed(3)}`);
                });
            }
        }

        console.log();
    });

    if (maxResults && results.length > maxResults) {
        console.log(`... and ${results.length - maxResults} more results`);
    }
}

/**
 * Save results to JSON file
 * @param {Object[]} results - Array of prediction results
 * @param {string} outputPath - Path to save the JSON file
 */
async function saveResults(results, outputPath) {
    try {
        await fs.writeFile(outputPath, JSON.stringify(results, null, 2));
        console.log(`Results saved to: ${outputPath}`);
    } catch (error) {
        console.error(`Error saving results: ${error.message}`);
    }
}

/**
 * Main function
 */
async function main() {
    // Setup command line arguments
    program
        .name('slide-classifier')
        .description('Slide Classification Inference using ONNX Runtime')
        .option('--model <path>', 'Path to ONNX model file', 'models/slide_classifier_mobilenetv4.onnx')
        .option('--model-info <path>', 'Path to model info JSON file', 'models/model_info.json')
        .option('--input <path>', 'Path to input image or directory', 'test')
        .option('--output <path>', 'Path to save results JSON file')
        .option('--confidence-threshold <number>', 'Minimum confidence threshold for predictions', parseFloat, 0.5)
        .option('--show-probabilities', 'Show probabilities for all classes', false)
        .option('--max-display <number>', 'Maximum number of detailed results to display (0 for all)', parseInt, 20)
        .option('--quiet', 'Only show summary, not detailed results', false);

    program.parse();
    const options = program.opts();

    try {
        // Initialize classifier
        console.log('Initializing slide classifier...');
        const classifier = new SlideClassifier(
            options.model,
            options.modelInfo,
            options.confidenceThreshold
        );

        await classifier.loadModel();

        // Determine input type and make predictions
        const inputPath = options.input;
        const startTime = Date.now();

        let results;
        const stats = await fs.stat(inputPath);

        if (stats.isFile()) {
            console.log(`Processing single image: ${inputPath}`);
            const result = await classifier.predictSingle(inputPath, options.showProbabilities);
            results = [result];
        } else if (stats.isDirectory()) {
            console.log(`Processing directory: ${inputPath}`);
            results = await classifier.predictDirectory(inputPath, options.showProbabilities);
        } else {
            console.error(`Input path is neither file nor directory: ${inputPath}`);
            process.exit(1);
        }

        const processingTime = (Date.now() - startTime) / 1000;

        if (!results || results.length === 0) {
            console.log('No images processed');
            return;
        }

        // Display results
        printResultsSummary(results);

        if (!options.quiet) {
            const maxDisplay = options.maxDisplay > 0 ? options.maxDisplay : null;
            printDetailedResults(results, options.showProbabilities, maxDisplay);
        }

        console.log(`\nProcessing completed in ${processingTime.toFixed(2)} seconds`);
        console.log(`Average time per image: ${(processingTime / results.length).toFixed(3)} seconds`);

        // Save results if requested
        if (options.output) {
            await saveResults(results, options.output);
        }

    } catch (error) {
        console.error(`Error: ${error.message}`);
        process.exit(1);
    }
}

// Run the main function if this script is executed directly
if (require.main === module) {
    main().catch(error => {
        console.error(`Unhandled error: ${error.message}`);
        process.exit(1);
    });
}

module.exports = { SlideClassifier };