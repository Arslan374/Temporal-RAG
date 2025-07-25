from crag_rag.time_aware import TimeAwareModule

def main():
    module = TimeAwareModule()
    print("\n--- Testing Temporal Score ---")
    print(f"'What happened in 2008?' Score: {module.get_temporal_score('What happened in 2008?')}")
    print(f"'Who is the president?' Score: {module.get_temporal_score('Who is the president?')}")
    print(f"'Events after March 2020?' Score: {module.get_temporal_score('Events after March 2020?')}")
    print("\n--- Testing Temporal Query Classification ---")
    print(f"'What happened in 2008?' Is temporal: {module.is_temporal_query('What happened in 2008?')}")
    print(f"'Who is the president?' Is temporal: {module.is_temporal_query('Who is the president?')}")
    print(f"'Events after March 2020?' Is temporal: {module.is_temporal_query('Events after March 2020?', threshold=0.2)}")
    print("\n--- Testing Timestamp Relevance ---")
    print(f"Query 2020-01-15, Doc 2020-01-20: {module.get_temporal_relevance_from_timestamps('2020-01-15', '2020-01-20')}")
    print(f"Query 2020-01-15, Doc 2021-01-15: {module.get_temporal_relevance_from_timestamps('2020-01-15', '2021-01-15')}")
    print(f"Query 2010, Doc 2020: {module.get_temporal_relevance_from_timestamps('2010', '2020')}")
    print(f"Query no-date, Doc 2020: {module.get_temporal_relevance_from_timestamps('', '2020')}")

if __name__ == '__main__':
    main()
