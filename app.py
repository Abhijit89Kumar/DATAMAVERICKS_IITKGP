#!/usr/bin/env python3
"""
HiLabs Contract Analysis Frontend
=================================

Flask web application for running and monitoring contract analysis pipeline.
Features:
- Clean, professional UI
- Real-time progress tracking via Server-Sent Events
- Structured output parsing
- Collapsible result sections
- Lightweight and responsive design

Author: AI Assistant
"""

import os
import sys
import json
import subprocess
import threading
import time
import queue
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from flask import Flask, render_template, request, jsonify, Response, stream_template
import logging

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Global variables for process management
current_process = None
output_queue = queue.Queue()
process_status = {
    'running': False,
    'stage': 'idle',
    'progress': 0,
    'start_time': None,
    'end_time': None,
    'results': {},
    'logs': []
}

class OutputParser:
    """Parse main_orchestrator.py output into structured sections."""
    
    @staticmethod
    def parse_line(line: str) -> Dict[str, Any]:
        """Parse a single output line and categorize it."""
        line = line.strip()
        if not line:
            return None
            
        # Stage detection patterns
        stage_patterns = {
            'initialization': r'Initializing.*Orchestrator|=' * 70,
            'finding_files': r'STEP 1: Finding Files|Finding.*contracts|Finding.*templates',
            'llm_demo': r'STEP 2: LLM Query Generation|Demonstrating.*LLM',
            'processing_templates': r'STEP 3: Processing Templates|Processing template',
            'processing_contracts': r'STEP 4: Processing Contracts|Processing contract',
            'classification': r'STEP 5: Classification Analysis|Running.*classification',
            'llm_explanations': r'STEP 6: LLM Explanations|Running.*explanations',
            'saving_results': r'STEP 8: Saving Results|Saving.*results|Creating.*report',
            'completion': r'SUCCESS!|Analysis completed|Results saved'
        }
        
        # Determine line type and stage
        line_type = 'info'
        stage = 'unknown'
        
        # Check for errors
        if any(keyword in line.lower() for keyword in ['error', 'failed', 'exception', 'traceback']):
            line_type = 'error'
        elif any(keyword in line.lower() for keyword in ['warning', 'warn']):
            line_type = 'warning'
        elif any(keyword in line.lower() for keyword in ['success', 'completed', 'saved']):
            line_type = 'success'
        elif line.startswith('STEP') or '=' in line:
            line_type = 'header'
            
        # Determine stage
        for stage_name, pattern in stage_patterns.items():
            if re.search(pattern, line, re.IGNORECASE):
                stage = stage_name
                break
                
        # Extract numbers/progress if present
        numbers = re.findall(r'\d+', line)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'line': line,
            'type': line_type,
            'stage': stage,
            'numbers': numbers
        }
    
    @staticmethod
    def estimate_progress(stage: str, line: str) -> int:
        """Estimate overall progress based on current stage and line content."""
        stage_weights = {
            'initialization': 5,
            'finding_files': 10,
            'llm_demo': 15,
            'processing_templates': 40,
            'processing_contracts': 70,
            'classification': 85,
            'llm_explanations': 95,
            'saving_results': 98,
            'completion': 100
        }
        
        base_progress = stage_weights.get(stage, 0)
        
        # Fine-tune based on line content
        if 'processing' in line.lower() and '/' in line:
            # Try to extract current/total from lines like "Processing 3/10"
            numbers = re.findall(r'(\d+)/?(\d+)?', line)
            if numbers and len(numbers[0]) == 2:
                current, total = numbers[0]
                if total and int(total) > 0:
                    sub_progress = (int(current) / int(total)) * 10
                    return min(base_progress + int(sub_progress), 100)
        
        return base_progress

def run_analysis():
    """Run the main_orchestrator.py script and capture output."""
    global current_process, process_status
    
    try:
        process_status.update({
            'running': True,
            'stage': 'initialization',
            'progress': 0,
            'start_time': datetime.now(),
            'end_time': None,
            'results': {},
            'logs': []
        })
        
        # Start the main_orchestrator.py process
        cmd = [sys.executable, 'main_orchestrator.py']
        current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Read output line by line
        while True:
            line = current_process.stdout.readline()
            if not line and current_process.poll() is not None:
                break
                
            if line:
                # Parse the line
                parsed = OutputParser.parse_line(line)
                if parsed:
                    # Update process status
                    if parsed['stage'] != 'unknown':
                        process_status['stage'] = parsed['stage']
                    
                    process_status['progress'] = OutputParser.estimate_progress(
                        process_status['stage'], parsed['line']
                    )
                    
                    process_status['logs'].append(parsed)
                    
                    # Put in queue for SSE
                    output_queue.put({
                        'type': 'progress',
                        'data': {
                            'stage': process_status['stage'],
                            'progress': process_status['progress'],
                            'line': parsed['line'],
                            'line_type': parsed['type'],
                            'timestamp': parsed['timestamp']
                        }
                    })
        
        # Process completed
        return_code = current_process.wait()
        process_status.update({
            'running': False,
            'end_time': datetime.now(),
            'progress': 100 if return_code == 0 else process_status['progress']
        })
        
        # Load results BEFORE sending completion signal
        if return_code == 0:
            load_analysis_results()
        
        # Send completion signal AFTER loading results
        output_queue.put({
            'type': 'complete',
            'data': {
                'return_code': return_code,
                'end_time': process_status['end_time'].isoformat(),
                'success': return_code == 0
            }
        })
            
    except Exception as e:
        process_status.update({
            'running': False,
            'end_time': datetime.now()
        })
        output_queue.put({
            'type': 'error',
            'data': {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        })

def load_analysis_results():
    """Load the latest analysis results and structure them for the dashboard."""
    try:
        # Find the latest analysis directory
        analysis_dir = Path('contract_analysis_results')
        if not analysis_dir.exists():
            return
            
        # Get the most recent analysis folder
        analysis_folders = [d for d in analysis_dir.iterdir() if d.is_dir() and d.name.startswith('analysis_')]
        if not analysis_folders:
            return
            
        latest_folder = max(analysis_folders, key=lambda x: x.stat().st_mtime)
        
        # Load comprehensive analysis summary
        comprehensive_summary = {}
        comprehensive_file = latest_folder / 'comprehensive_analysis_summary.json'
        if comprehensive_file.exists():
            with open(comprehensive_file, 'r') as f:
                comprehensive_summary = json.load(f)
        
        # Load classification analysis
        classification_data = {}
        classification_file = latest_folder / 'classification_analysis.json'
        if classification_file.exists():
            with open(classification_file, 'r') as f:
                classification_data = json.load(f)
        
        # Load LLM explanations
        llm_explanations = {}
        explanations_file = latest_folder / 'llm_explanations' / 'non_standard_explanations.json'
        if explanations_file.exists():
            with open(explanations_file, 'r') as f:
                llm_explanations = json.load(f)
        
        # Structure results for dashboard
        results = {
            'session_info': comprehensive_summary.get('session_info', {}),
            'contract_results': comprehensive_summary.get('contract_results', []),
            'template_results': comprehensive_summary.get('template_results', []),
            'classification_results': classification_data.get('classification_results', []),
            'llm_explanations': llm_explanations,
            'output_directory': str(latest_folder)
        }
        
        # Calculate analytics for dashboard
        analytics = calculate_dashboard_analytics(results)
        results['analytics'] = analytics
        
        process_status['results'] = results
        
        # Send results to frontend
        output_queue.put({
            'type': 'results',
            'data': results
        })
        
    except Exception as e:
        app.logger.error(f"Failed to load analysis results: {e}")

def calculate_dashboard_analytics(results):
    """Calculate analytics from the loaded results for dashboard display."""
    analytics = {
        'total_clauses': 0,
        'standard_clauses': 0,
        'non_standard_clauses': 0,
        'total_contracts': 0,
        'successful_contracts': 0,
        'total_time': 0,
        'tn_contracts': 0,
        'wa_contracts': 0,
        'tn_standard': 0,
        'tn_non_standard': 0,
        'wa_standard': 0,
        'wa_non_standard': 0,
        'similarity_scores': {
            'semantic': [],
            'cosine': [],
            'jaccard': []
        }
    }
    
    # Process session info
    session_info = results.get('session_info', {})
    analytics['total_contracts'] = session_info.get('total_contracts', 0)
    analytics['successful_contracts'] = session_info.get('successful_contracts', 0)
    
    # Process contract results for timing
    contract_results = results.get('contract_results', [])
    if contract_results:
        total_time = sum(c.get('processing_time', 0) for c in contract_results)
        analytics['total_time'] = round(total_time, 2)
    
    # Process classification results
    classification_results = results.get('classification_results', [])
    contract_clauses = {}  # Track clauses per contract for risk analysis
    
    for result in classification_results:
        analytics['total_clauses'] += 1
        
        # Count standard vs non-standard
        if result.get('classification') == 'standard':
            analytics['standard_clauses'] += 1
        else:
            analytics['non_standard_clauses'] += 1
        
        # Count by contract type
        contract_type = result.get('contract_type', '')
        if 'TN' in contract_type:
            analytics['tn_contracts'] = max(analytics['tn_contracts'], 1)  # At least 1 if we see TN clauses
            if result.get('classification') == 'standard':
                analytics['tn_standard'] += 1
            else:
                analytics['tn_non_standard'] += 1
        elif 'WA' in contract_type:
            analytics['wa_contracts'] = max(analytics['wa_contracts'], 1)  # At least 1 if we see WA clauses
            if result.get('classification') == 'standard':
                analytics['wa_standard'] += 1
            else:
                analytics['wa_non_standard'] += 1
        
        # Collect similarity scores for averages
        if result.get('semantic_similarity') is not None:
            analytics['similarity_scores']['semantic'].append(result.get('semantic_similarity'))
        if result.get('cosine') is not None:
            analytics['similarity_scores']['cosine'].append(result.get('cosine'))
        if result.get('jaccard') is not None:
            analytics['similarity_scores']['jaccard'].append(result.get('jaccard'))
        
        # Track clauses per contract for risk analysis
        contract_clause_key = result.get('contract_type', 'unknown')
        if contract_clause_key not in contract_clauses:
            contract_clauses[contract_clause_key] = {'total': 0, 'non_standard': 0}
        contract_clauses[contract_clause_key]['total'] += 1
        if result.get('classification') != 'standard':
            contract_clauses[contract_clause_key]['non_standard'] += 1
    
    # Calculate percentages
    if analytics['total_clauses'] > 0:
        analytics['standard_percentage'] = round((analytics['standard_clauses'] / analytics['total_clauses']) * 100, 1)
        analytics['non_standard_percentage'] = round(100 - analytics['standard_percentage'], 1)
    else:
        analytics['standard_percentage'] = 0
        analytics['non_standard_percentage'] = 0
    
    # Calculate success rate
    if analytics['total_contracts'] > 0:
        analytics['success_rate'] = round((analytics['successful_contracts'] / analytics['total_contracts']) * 100, 1)
        analytics['avg_time_per_contract'] = round(analytics['total_time'] / analytics['total_contracts'], 2)
    else:
        analytics['success_rate'] = 0
        analytics['avg_time_per_contract'] = 0
    
    # Calculate average similarity scores
    for score_type, scores in analytics['similarity_scores'].items():
        if scores:
            analytics[f'avg_{score_type}'] = round(sum(scores) / len(scores), 3)
        else:
            analytics[f'avg_{score_type}'] = 0
    
    
    # Count from actual distinct contract files for more accurate counts
    tn_count = len([c for c in contract_results if 'TN_' in c.get('contract', '')])
    wa_count = len([c for c in contract_results if 'WA_' in c.get('contract', '')])
    analytics['tn_contracts'] = tn_count
    analytics['wa_contracts'] = wa_count
    
    # Process LLM explanations
    llm_explanations = results.get('llm_explanations', {})
    analytics['llm_explanations_count'] = len(llm_explanations.get('explanations', [])) if isinstance(llm_explanations.get('explanations'), list) else 0
    
    return analytics

@app.route('/')
def index():
    """Main page route."""
    return render_template('index.html')

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    """Start the contract analysis process."""
    global process_status
    
    if process_status['running']:
        return jsonify({'error': 'Analysis is already running'}), 400
    
    # Start analysis in background thread
    thread = threading.Thread(target=run_analysis)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started', 'message': 'Analysis started successfully'})

@app.route('/stop_analysis', methods=['POST'])
def stop_analysis():
    """Stop the running analysis process."""
    global current_process, process_status
    
    if current_process and current_process.poll() is None:
        current_process.terminate()
        current_process = None
    
    process_status.update({
        'running': False,
        'end_time': datetime.now()
    })
    
    return jsonify({'status': 'stopped', 'message': 'Analysis stopped'})

@app.route('/status')
def get_status():
    """Get current process status."""
    return jsonify(process_status)

@app.route('/stream')
def stream():
    """Server-Sent Events stream for real-time updates."""
    def event_stream():
        while True:
            try:
                # Get data from queue with timeout
                data = output_queue.get(timeout=1.0)
                yield f"data: {json.dumps(data)}\n\n"
                output_queue.task_done()
            except queue.Empty:
                # Send heartbeat
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'data': {'error': str(e)}})}\n\n"
                break
    
    return Response(event_stream(), mimetype="text/event-stream")

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("Starting HiLabs Contract Analysis Frontend")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host='localhost', port=5000)
