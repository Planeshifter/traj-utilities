'use strict';

// MODULES //

var cast = require( 'dstructs-cast-arrays' );
var divide = require( 'compute-divide' );
var dot = require( 'compute-dot' );
var exp = require( 'math-exp' );
var ln = require( 'math-ln' );
var pdf = require( 'distributions-truncated-normal-pdf' );
var pow = require( 'math-power' );
var sum = require( 'compute-sum' );
var mean = require( 'compute-mean' );
var toMatrix = require( 'compute-to-matrix' );


// MODEL //

class Model {

	constructor( nOutcomes, nGroups, opts ) {
		// Number of Responses:
		this.nOutcomes = nOutcomes;
		// Number of Groups:
		this.nGroups = nGroups;

		if ( opts ) {
			// Multi-risk coefficients:
			this.theta = opts.theta;
			// Polynomial coefficients:
			this.beta = opts.beta;
			// Estimated standard deviations:
			this.sigma = opts.sigma;
			// Time variable:
			this.times = opts.times;
			// Endpoints of support for censored normal likelihood:
			this.limits = opts.limits;
		}
	}

	addIntercept( x ) {
		x.unshift( 1 );
	}

	getFittedLine( group, outcome ) {
		let betas = this.beta[group][outcome];
		let fittedValues = new Array( this.times.length );
		for ( let i = 0; i < this.times.length; i++ ) {
			let mean = betas[ 0 ];
			if ( betas.length > 1 ) {
				let j = 1;
				while ( j < betas.length ) {
					mean += betas[j] * pow( this.times[ i ], j );
					j++;
				}
			}
			fittedValues[ i ] = mean;
		}
		return fittedValues;
	}

	getCollapsedLines( outcome, groupings ) {
		let ngroups = groupings.length;
		let ret = {};
		for ( let i = 0; i < ngroups; i++ ) {
			let group = groupings[ i ];
			let lines = new Array( group.length );
			for ( let j = 0; j < group.length; j++ ) {
				let member = group[ j ];
				lines[ j ] = this.getFittedLine( member, outcome );
			}
			lines = toMatrix( lines );
			if ( lines.shape[0] > 1 ) {
				ret[ i ] = cast( mean( lines, { 'dim': 1 } ).data, 'generic' );
			} else {
				ret[ i ] = cast( lines.data, 'generic' );
			}
		}
		return ret;
	}

	linearPredictor( x, group ) {
		return dot( x, this.theta[ group ] );
	}

	conditionalLogLikelihood( vals, group, outcome ) {
		let nPeriods = vals.length;
		let logLik = 0;
		for ( let i = 0; i < nPeriods; i++ ) {
			if ( vals[ i ] !== null ) {
				let betas = this.beta[group][outcome];
				let time = this.times[ i ];
				let mean = betas[ 0 ];
				if ( betas.length > 1 ) {
					let j = 1;
					while ( j < betas.length ) {
						mean += betas[j] * pow( time, j );
						j++;
					}
				}
				logLik += ln( pdf( vals[ i ], {
					mu: mean,
					sigma: this.sigma[ outcome ],
					a: this.limits[ outcome ][ 0 ],
					b: this.limits[ outcome ][ 1 ]
				} ) );
			}
		}
		return logLik;
	}

	posteriorProbabilities( y, x ) {
		let probs = new Array( this.nGroups );
		let risks = this.softmax( x );
		let logLiks = new Array( this.nGroups );
		for( let i = 0; i < this.nGroups; i++ ) {
			logLiks[ i ] = ln( risks[ i ] );
			for ( let j = 0; j < this.nOutcomes; j++ ) {
				logLiks[ i ] += this.conditionalLogLikelihood( y[ j ], i, j );
			}
			probs[ i ] = exp( logLiks[ i ] );
		}

		let normalizingConstant = sum( probs );
		probs = divide( probs, normalizingConstant );
		return probs;
	}

	softmax( x ) {
		// Make shallow copy of input array:
		x = x.slice();
		console.log( x )
		// Add intercept dummy:
		this.addIntercept( x );
		let groupSum = 0;
		let groupScores = new Array( this.nGroups );
		for ( let i = 0; i < this.nGroups; i++ ) {
			let score = exp( this.linearPredictor( x, i ) );
			groupScores[ i ] = score;
			groupSum += score;
		}
		for ( let i = 0; i < this.nGroups; i++ ) {
			groupScores[ i ] = groupScores[ i ] / groupSum;
		}
		return groupScores;
	}
}


// EXPORTS //

module.exports = Model;
