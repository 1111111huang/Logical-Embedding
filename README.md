# Logical-Embedding
Word embedding research project with G.Penn

**dataset.py:**

    Partial order data sets:    create the data .txt files for some partial order.
    reslipt(train, facts, no_link_percent) -- ommited, resplit the train and fact data
    Data class:     seed -- random seed
                    type_check -- omitted ???
                    domain_size -- default 128, NOT USED
                    use_extra_facts -- omitted ???
                    query_include_reverse -- set to True ???
                    relation_to_number -- number the relations
                    entity_to_number -- number the entities
                    number_to_entity -- map back from numbers to entities
                    num_relation -- number of relations
                    num_query -- twice the number of relations since the inv_ realtions
                    num_entity -- number of entities
                    matrix_db -- records like {relation_index: [[0,0],[head,tail],...],[0,1,1,...],[self.num_entity, self.num_entity]}
                    fact,train,test,valid -- [(relation #, entity_1#, entity_2#) ...]
                    parser -- {"query": {relation#: relation_name}, "operators": {relation_name: relation#}}
                    domains -- None ???
                    num_operator -- 2* number of relations
                    
                    create_parser(self) -- initialize the parser
                    _numerical_encode(self) -- numbering the relations and entities
                    _parse_triplets(self, file) -- convert the (entity1, relation, entity2) -> (entity1#, relation, entity2#)
                    count_batch, reset, next_train, next_test, next_test, next_batch -- omited due to batch related
    
    
