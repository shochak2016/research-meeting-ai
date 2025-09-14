"""
Context management for LLM queries.
Handles formatting and organizing different types of context:
1. Meeting transcriptions
2. Studies from vector database
3. Past queries
4. Past answers
"""

def track_previous_studies(current_studies, previous_studies):
    """
    Mark studies that were previously retrieved to avoid re-querying.
    
    Args:
        current_studies: List of study dicts from current query
        previous_studies: List of study dicts from previous queries
    
    Returns:
        Combined list with previously retrieved studies marked
    """
    if not previous_studies:
        return current_studies
    
    # Create a set of PMIDs from previous studies for fast lookup
    previous_pmids = {study.get('pmid') for study in previous_studies if study.get('pmid')}
    previous_titles = {study.get('title') for study in previous_studies if study.get('title')}
    
    # Mark current studies if they were seen before
    marked_studies = []
    for study in current_studies:
        study_copy = study.copy()
        pmid = study_copy.get('pmid')
        title = study_copy.get('title')
        
        # Check if this study was previously retrieved
        if (pmid and pmid in previous_pmids) or (title and title in previous_titles):
            study_copy['previously_retrieved'] = True
        
        marked_studies.append(study_copy)
    
    return marked_studies


def create_context(transcription=None, studies=None, past_queries=None, past_answers=None, previous_studies=None, 
                  max_chars=2000, priority_order=['transcription', 'studies', 'queries', 'answers']):
    """
    Create formatted context from different sources.
    
    Args:
        transcription: Meeting transcription text
        studies: List of dicts with study metadata (title, abstract, pmid, etc.)
        past_queries: List of previous user queries
        past_answers: List of previous LLM answers
        previous_studies: List of previously retrieved studies to mark as cached
        max_chars: Max characters per section to avoid token limits
        priority_order: List/tuple defining importance order, e.g. ['transcription', 'studies', 'queries', 'answers']
                       Items listed first are kept when truncating. Default: ['transcription', 'studies', 'queries', 'answers']
    
    Returns:
        Formatted context string
    """
    
    context_sections = {}
    
    # Add meeting transcription
    if transcription:
        transcript_text = transcription[:max_chars] if len(transcription) > max_chars else transcription
        context_sections['transcription'] = f"=== MEETING TRANSCRIPTION ===\n{transcript_text}\n"
    
    # Add studies from vector DB
    if studies:
        # Mark any previously retrieved studies
        if previous_studies:
            studies = track_previous_studies(studies, previous_studies)
        
        studies_text = "=== RELEVANT STUDIES ===\n"
        for i, study in enumerate(studies[:5], 1):  # Limit to top 5 studies
            title = study.get('title', 'Untitled')
            abstract = study.get('abstract', '')[:300]  # First 300 chars of abstract
            pmid = study.get('pmid', '')
            authors = study.get('authors', '')
            score = study.get('_score', '')
            
            # Mark if previously retrieved
            prefix = "[CACHED] " if study.get('previously_retrieved') else ""
            studies_text += f"[{i}] {prefix}{title}\n"
            if authors:
                studies_text += f"    Authors: {authors}\n"
            if pmid:
                studies_text += f"    PMID: {pmid}\n"
            if score:
                studies_text += f"    Relevance: {score:.3f}\n"
            if abstract:
                studies_text += f"    Abstract: {abstract}...\n"
            studies_text += "\n"
        context_sections['studies'] = studies_text
    
    # Add past queries
    if past_queries:
        queries_text = "=== PREVIOUS QUERIES ===\n"
        # Get last 5 queries
        for query in past_queries[-5:]:
            queries_text += f"• {query}\n"
        context_sections['queries'] = queries_text
    
    # Add past answers (abbreviated)
    if past_answers:
        answers_text = "=== PREVIOUS ANSWERS (Summary) ===\n"
        # Get last 3 answers, abbreviated
        for answer in past_answers[-3:]:
            abbreviated = answer[:200] + "..." if len(answer) > 200 else answer
            answers_text += f"• {abbreviated}\n\n"
        context_sections['answers'] = answers_text
    
    # Build context based on priority order
    context_parts = []
    total_chars = 0
    max_total_chars = max_chars * 4  # Total context limit
    
    for section_key in priority_order:
        if section_key in context_sections:
            section_text = context_sections[section_key]
            # Check if adding this section would exceed limit
            if total_chars + len(section_text) <= max_total_chars:
                context_parts.append(section_text)
                total_chars += len(section_text)
            else:
                # Add truncated version if there's room
                remaining = max_total_chars - total_chars
                if remaining > 100:  # Only add if meaningful
                    truncated = section_text[:remaining-50] + "\n[Section truncated]\n"
                    context_parts.append(truncated)
                break  # Stop adding sections
    
    full_context = "\n".join(context_parts)
    return full_context


def create_conversation_context(messages, max_messages=10):
    """
    Create context from conversation history.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        max_messages: Maximum number of messages to include
    
    Returns:
        Formatted conversation context
    """
    if not messages:
        return ""
    
    context = "=== CONVERSATION HISTORY ===\n"
    recent_messages = messages[-max_messages:]
    
    for msg in recent_messages:
        role = msg.get('role', 'unknown').upper()
        content = msg.get('content', '')
        # Truncate long messages
        if len(content) > 500:
            content = content[:500] + "..."
        context += f"{role}: {content}\n\n"
    
    return context


def prioritize_context(full_context, priority_order=None, max_chars=8000):
    """
    Prioritize context sections if total exceeds limit.
    
    Args:
        full_context: Complete context string with sections
        priority_order: List of section names in priority order
        max_chars: Maximum total characters
    
    Returns:
        Prioritized context within limits
    """
    if len(full_context) <= max_chars:
        return full_context
    
    if not priority_order:
        # Default priority: transcription > studies > queries > answers
        priority_order = ["MEETING TRANSCRIPTION", "RELEVANT STUDIES", "PREVIOUS QUERIES", "PREVIOUS ANSWERS"]
    
    sections = {}
    current_section = None
    current_content = []
    
    # Parse sections
    for line in full_context.split('\n'):
        if line.startswith("==="):
            if current_section:
                sections[current_section] = '\n'.join(current_content)
            current_section = line.strip('= ')
            current_content = [line]
        else:
            current_content.append(line)
    
    # Add last section
    if current_section:
        sections[current_section] = '\n'.join(current_content)
    
    # Rebuild with priority
    prioritized = []
    char_count = 0
    
    for section_name in priority_order:
        if section_name in sections:
            section_text = sections[section_name]
            if char_count + len(section_text) <= max_chars:
                prioritized.append(section_text)
                char_count += len(section_text)
            else:
                # Add truncated version
                remaining = max_chars - char_count
                if remaining > 100:  # Only add if meaningful amount remains
                    truncated = section_text[:remaining-50] + "\n[Section truncated]"
                    prioritized.append(truncated)
                break
    
    return '\n\n'.join(prioritized)


