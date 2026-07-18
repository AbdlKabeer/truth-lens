import datetime

# Example: Metadata validation
def validate_metadata(article_metadata):
    # Check if the source is credible (e.g., published by reputable organizations)
    credible_sources = ['The Guardian Nigeria', 'Vanguard', 'Premium Times']
    
    if article_metadata['source'] not in credible_sources:
        return False
    
    # Check if the publication date is within a reasonable range (e.g., not too old)
    publication_date = datetime.datetime.strptime(article_metadata['date'], "%Y-%m-%d")
    current_date = datetime.datetime.now()
    
    if (current_date - publication_date).days > 365:  # Example: more than 1 year old
        return False
    
    return True
