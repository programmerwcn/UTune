import re
import logging

import psycopg2

from .database_connector import DatabaseConnector


class PostgresDatabaseConnector(DatabaseConnector):
    def __init__(self, config, autocommit=False, host=None,
                 port=None, db_name=None, user=None, password=None):
        DatabaseConnector.__init__(self, config, autocommit=autocommit)

        self.db_system = "postgres"
        self._connection = None

        if host is not None:
            self.host = host
        else:
            self.host = config["postgresql"]["host"]
        if port is not None:
            self.port = port
        else:
            self.port = config["postgresql"]["port"]
        if db_name is not None:
            self.db_name = db_name
        else:
            self.db_name = config["postgresql"]["database"]

        if user is not None:
            self.user = user
        else:
            self.user = config["postgresql"]["user"]
        if password is not None:
            self.password = password
        else:
            self.password = config["postgresql"]["password"]

        self.create_connection()

        # Set the random seed to obtain deterministic statistics
        self.set_random_seed()

        # logging.disable(logging.DEBUG)
        # logging.info("Postgres connector created: {}({})".format(self.db_name, self.host))
        # logging.disable(logging.INFO)

    def create_connection(self):
        if self._connection:
            self.close()

        self._connection = psycopg2.connect(host=self.host,
                                            database=self.db_name,
                                            port=self.port,
                                            user=self.user,
                                            password=self.password)
        self._connection.autocommit = self.autocommit
        self._cursor = self._connection.cursor()
        logging.debug("Database connector created: {} on {}".format(self.db_name, self.host))

    def set_random_seed(self, value=0.17):
        logging.info(f"Postgres: Set random seed `SELECT setseed({value})`")
        self.exec_only(f"SELECT setseed({value})")

    def enable_simulation(self):
        self.exec_only("create extension hypopg")
        self.commit()

    def database_names(self):
        result = self.exec_fetch("select datname from pg_database", False)
        return [x[0] for x in result]

    # Updates query syntax to work in PostgreSQL
    def update_query_text(self, text):
        text = text.replace(";\nlimit ", " limit ").replace("limit -1", "")
        text = re.sub(r" ([0-9]+) days\)", r" interval '\1 days')", text)
        text = self._add_alias_subquery(text)

        return text

    # PostgreSQL requires an alias for subqueries
    def _add_alias_subquery(self, query_text):
        text = query_text.lower()
        positions = []
        for match in re.finditer(r"((from)|,)[  \n]*\(", text):
            counter = 1
            pos = match.span()[1]
            while counter > 0:
                char = text[pos]
                if char == "(":
                    counter += 1
                elif char == ")":
                    counter -= 1
                pos += 1
            next_word = query_text[pos:].lstrip().split(" ")[0].split("\n")[0]
            if next_word[0] in [")", ","] or next_word in [
                "limit",
                "group",
                "order",
                "where",
            ]:
                positions.append(pos)
        for pos in sorted(positions, reverse=True):
            query_text = query_text[:pos] + " as alias123 " + query_text[pos:]

        return query_text

    def create_database(self, database_name):
        self.exec_only("create database {}".format(database_name))
        logging.info("Database {} created".format(database_name))

    def import_data(self, table, path, delimiter="|"):
        with open(path, "r") as file:
            self._cursor.copy_from(file, table, sep=delimiter, null="")

    def indexes_size(self):
        # Returns size in bytes
        statement = (
            "select sum(pg_indexes_size(table_name::text)) from "
            "(select table_name from information_schema.tables "
            "where table_schema='public') as all_tables"
        )
        result = self.exec_fetch(statement)

        return result[0]

    def drop_database(self, database_name):
        statement = f"DROP DATABASE {database_name};"
        self.exec_only(statement)

        logging.info(f"Database {database_name} dropped")

    def create_statistics(self):
        logging.info("Postgres: Run `analyze`")
        self.commit()
        self._connection.autocommit = True
        # : blocked?
        # self.exec_only("analyze")
        self._connection.autocommit = self.autocommit

    def supports_index_simulation(self):
        if self.db_system == "postgres":
            return True
        return False

    def _simulate_index(self, index):
        table_name = index.table()
        statement = (
            "select * from hypopg_create_index("
            f"'create index on {table_name} "
            f"({index.joined_column_names()})')"
        )

        # (0415): newly added. for column_name = keyword
        if "group" in statement:
            statement = statement.replace("(group)", "(\"group\")")
            statement = statement.replace("(group,", "(\"group\",")
            statement = statement.replace(",group,", ",\"group\",")
            statement = statement.replace(",group)", "\"group\")")

        result = self.exec_fetch(statement)

        return result

    def _drop_simulated_index(self, oid):
        statement = f"select * from hypopg_drop_index({oid})"
        result = self.exec_fetch(statement)

        assert result[0] is True, f"Could not drop simulated index with oid = {oid}."

    def create_index(self, index):
        table_name = index.table()
        statement = (
            f"create index {index.index_idx()} "
            f"on {table_name} ({index.joined_column_names()})"
        )
        self.exec_only(statement)
        size = self.exec_fetch(
            f"select relpages from pg_class c " f"where c.relname = '{index.index_idx()}'"
        )
        size = size[0]
        index.estimated_size = size * 8 * 1024

    def create_indexes(self, indexes, mode="hypo"):
        """
        :param mode: 'hypo' or not
        :param indexes: table#col1,col2#col1,col2,col3
        :return:
        """
        try:
            for index in indexes:
                index_def = index.split("#")
                index_name = index.replace("#", "_").replace(",", "_")
                stmt = f"create index {index_name} on {index_def[0]} ({index_def[1]})"
                if len(index_def) == 3:
                    stmt += f" include ({index_def[2]});"
                else:
                    stmt += ";"
                if mode == "hypo":
                    stmt = f"select * from hypopg_create_index('{stmt}')"
                self.exec_only(stmt)
                if mode != "hypo":
                    self.commit()
                    logging.info(f"Index {index_name} created")
        except Exception as e:
            print(e)
            print(stmt)

    def get_ind_cost(self, query, indexes, mode="hypo"):
        self.create_indexes(indexes, mode)

        stmt = f"explain (format json) {query}"
        query_plan = self.exec_fetch(stmt)[0][0]["Plan"]
        # drop view
        # self._cleanup_query(query)
        total_cost = query_plan["Total Cost"]

        if mode == "hypo":
            self.drop_hypo_indexes()
        else:
            self.drop_indexes()

        return total_cost

    def get_ind_plan(self, query, indexes, mode="hypo"):
        self.create_indexes(indexes, mode)

        stmt = f"explain (format json) {query}"
        query_plan = self.exec_fetch(stmt)[0][0]["Plan"]
        # drop view
        # self._cleanup_query(query)
        total_cost = query_plan["Total Cost"]

        if mode == "hypo":
            self.drop_hypo_indexes()
        else:
            self.drop_indexes()

        return query_plan
    
    def drop_hypo_indexes(self):
        # logging.info("Dropping hypo indexes")
        stmt = "SELECT * FROM hypopg_reset();"
        self.exec_only(stmt)

    def drop_indexes(self):
        # logging.info("Dropping indexes")
        stmt = "select indexname from pg_indexes where schemaname='public'"
        indexes = self.exec_fetch(stmt, one=False)
        for index in indexes:
            index_name = index[0]
            # (0408): newly added for real.
            if "_pkey" not in index_name and "primary" not in index_name:
                drop_stmt = "drop index {}".format(index_name)
                logging.debug("Dropping index {}".format(index_name))
                self.exec_only(drop_stmt)
    
    def drop_chosen_indexes(self, indexes):
        for index in indexes:
            index_def = index.split("#")
            column_names = index_def[1].split(",")
            # index_name = index.replace("#", "_").replace(",", "_")
            stmt = f"SELECT indexname FROM pg_indexes WHERE tablename = \'{index_def[0]}\' AND indexdef LIKE \'%({', '.join(column_names)})%\';"
            self._cursor.execute(stmt)
            candidate_indexes = self._cursor.fetchall()
            for candidate_index in candidate_indexes:
                index_name = candidate_index[0]
                if "_pkey" not in index_name and "primary" not in index_name:
                    self._cursor.execute(f"DROP INDEX IF EXISTS {index_name}")
                    self.commit()

    # PostgreSQL expects the timeout in milliseconds
    def exec_query(self, query, timeout=None, cost_evaluation=False):
        # Committing to not lose indexes after timeout
        if not cost_evaluation:
            self._connection.commit()
        query_text = self._prepare_query(query)
        if timeout:
            set_timeout = f"set statement_timeout={timeout}"
            self.exec_only(set_timeout)
        statement = f"explain (analyze, buffers, format json) {query_text}"
        try:
            plan = self.exec_fetch(statement, one=True)[0][0]["Plan"]
            result = plan["Actual Total Time"], plan
        except psycopg2.OperationalError as e:
        # Check if it's a timeout error
            if 'statement timeout' in str(e):
                print(f"Query timed out: {e}")
                # Optionally, rollback if you have started a transaction
                self._connection.rollback()
                return None, None  # or handle as needed

            # Handle other operational errors
            print(f"Operational error: {e}")
            self._connection.rollback()
            return None
        except Exception as e:
            logging.error(f" {e}")
            self._connection.rollback()
            result = None, self._get_plan(query)
            # exp_res = timeout, self._get_plan(query)
        # Disable timeout
        self._cursor.execute("set statement_timeout = 0")
        # drop view
        self._cleanup_query(query)

        return result
    
    def exec_query_txt(self, query_txt, timeout=None, cost_evaluation=False):
        if not cost_evaluation:
            self._connection.commit()
        query_text = self.prepare_views(query_txt)
        if timeout:
            set_timeout = f"set statement_timeout={timeout}"
            self.exec_only(set_timeout)
        statement = f"explain (analyze, buffers, format json) {query_text}"
        try:
            plan = self.exec_fetch(statement, one=True)[0][0]["Plan"]
            result = plan["Actual Total Time"], plan
        except Exception as e:
            logging.error(f"{query_txt}, {e}")
            self._connection.rollback()
            result = None, self.get_plan(query_txt)
            # exp_res = timeout, self._get_plan(query)
        # Disable timeout
        self._cursor.execute("set statement_timeout = 0")
        # drop view
        self.cleanup_views(query_txt)
        
        return result

    
    def exec_query_without_index(self, query, timeout=None, cost_evaluation=False):
        self.exec_only("set enable_indexscan = off;")
        # Committing to not lose indexes after timeout
        result = self.exec_query(query, timeout, cost_evaluation)
        self.exec_only("set enable_indexscan = on; ")
        return result

    def _cleanup_query(self, query):
        """
        Drop view created in the query
        :param query:
        :return:
        """
        for query_statement in query.text.split(";"):
            if "drop view" in query_statement:
                self.exec_only(query_statement)
                self.commit()
    
    def cleanup_views(self, query_txt):
        for query_statement in query_txt.split(";"):
            if "drop view" in query_statement:
                self.exec_only(query_statement)
                self.commit()

    def prepare_views(self, query_txt):
        if "create view" in query_txt:
            for query_statement in query_txt.split(";"):
                if "create view" in query_statement:
                    try:
                        self.exec_only(query_statement)
                    except Exception as e:
                        logging.error(e)
                elif "select" in query_statement or "SELECT" in query_statement:
                    return query_statement
        else:
            return query_txt

    def _get_cost(self, query):
        query_plan = self._get_plan(query)
        total_cost = query_plan["Total Cost"]

        return total_cost

    def _get_plan(self, query):
        # create view and return the next sql.
        query_text = self._prepare_query(query)
        statement = f"explain (format json) {query_text}"
        query_plan = self.exec_fetch(statement)[0][0]["Plan"]
        # drop view
        self._cleanup_query(query)

        return query_plan
    
    def get_plan(self, query_txt):
        query_txt = self.prepare_views(query_txt)
        statement = f"explain (format json) {query_txt}"
        query_plan = self.exec_fetch(statement)[0][0]["Plan"]
        self.cleanup_views(query_txt)
        return query_plan


    def number_of_indexes(self):
        statement = """select count(*) from pg_indexes
                       where schemaname = 'public'"""
        result = self.exec_fetch(statement)

        return result[0]

    def table_exists(self, table_name):
        statement = f"""SELECT EXISTS (
            SELECT 1
            FROM pg_tables
            WHERE tablename = '{table_name}');"""
        result = self.exec_fetch(statement)

        return result[0]

    def database_exists(self, database_name):
        statement = f"""SELECT EXISTS (
            SELECT 1
            FROM pg_database
            WHERE datname = '{database_name}');"""
        result = self.exec_fetch(statement)

        return result[0]

    def get_tables(self):
        tables = []
        sql = "select tablename from pg_tables where schemaname = 'public';"
        rows = self.exec_fetch(sql, one=False)
        for row in rows:
            tables.append(row[0])

        return tables

    def get_cols(self, table):
        cols = []
        sql = f"select column_name from information_schema.columns where " \
              f"table_schema='public' and table_name='{table}'"

        rows = self.exec_fetch(sql, one=False)
        for row in rows:
            cols.append(row[0])

        return cols
    
    def get_row_count(self, tables):
        rows = list()
        for table in tables:
            sql = f"select count(*) from {table}"
            count = self.exec_fetch(sql, one=True)
            rows.append((table, count[0]))
        return rows

    def get_null_frac(self, table, columns):
            null_frac = list()
            for column in columns:
                sql = f"select null_frac from pg_stats where tablename='{table}' and attname='{column}'"
                rows = self.exec_fetch(sql, one=True)
                if rows is None or rows[0] is None:  # Handle cases where no result is returned
                    null_frac.append(None)  # Append None or a default value
                else:
                    null_frac.append(round(rows[0], 4))
            return null_frac
        
    def get_dist_frac(self, table, columns):
        dist_frac = list()
        for column in columns:
            sql = f"select count(distinct {column}) * 1.0 / count(*) as distinct_ratio from {table}"
            rows = self.exec_fetch(sql, one=True)
            dist_frac.append(round(rows[0], 4))
        return dist_frac

