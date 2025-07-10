import sqlite3
import shutil
import os
from datetime import datetime
import json


def create_database():

  if not os.path.exists('my_DB.db'):
  
    # Connect to the SQLite database
    database_path = 'my_DB.db'
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Create a table named 'industry'
    cursor.execute('''
    CREATE TABLE Industry (
        IndustryID INTEGER PRIMARY KEY,
        IndustryName VARCHAR(100) UNIQUE)
    ''')

    # Create a table named 'client'
    cursor.execute('''
    CREATE TABLE Clients (
        ClientID INTEGER PRIMARY KEY,
        ClientName VARCHAR(100) UNIQUE,
        ContactEmail VARCHAR(50),
        ContactNumber VARCHAR(20),
        Location VARCHAR(50),
        IndustryID INTEGER,
        FOREIGN KEY (IndustryID) REFERENCES Industry(IndustryID))
        
    ''')

    # Create a table named 'project'
    cursor.execute('''
    CREATE TABLE Project (
        ProjectID INTEGER PRIMARY KEY,
        ClientID INTEGER,
        ProjectName VARCHAR(100),
        StartDate DATE,
        EndDate DATE,
        NumUsers INTEGER,
        ProjectStatus CHAR(10),
        Budget FLOAT,
        DeliveryModel VARCHAR(10),
        FOREIGN KEY (ClientID) REFERENCES Clients(ClientID))
    ''')

    # Create a table named 'TechnologyStack'
    cursor.execute('''
    CREATE TABLE TechnologyStack (
        TechID INTEGER PRIMARY KEY,
        TechName VARCHAR(100),
        Category VARCHAR(100)
        )
    ''')

    # Create a table named 'ProjectTechnology'
    cursor.execute('''
    CREATE TABLE ProjectTechnology (
        ProjectID INTEGER,
        TechID INTEGER,
        Status VARCHAR(15),
        PRIMARY KEY (ProjectID, TechID),
        FOREIGN KEY (ProjectID) REFERENCES Project(ProjectID),
        FOREIGN KEY (TechID) REFERENCES TechnologyStack(TechID))
    ''')

    # Create a lookup table named 'SourceType'
    cursor.execute('''
    CREATE TABLE SourceType (
        SourceTypeID INTEGER PRIMARY KEY,
        SourceTypeName VARCHAR(50) UNIQUE)
    ''')

    # Create a table named 'InteractionLog'
    cursor.execute('''
    CREATE TABLE InteractionLog (
        InteractionID INTEGER PRIMARY KEY,
        Timestamp DATETIME,
        SourceTypeID INTEGER,
        RawText VARCHAR,
        ExtractedSummary VARCHAR,
        FOREIGN KEY (SourceTypeID) REFERENCES SourceType(SourceTypeID)
    )
    ''')  

    # Create a lookup table named 'ConstraintType'
    cursor.execute('''
    CREATE TABLE ConstraintType (
        ConstraintTypeID INTEGER PRIMARY KEY,
        ConstraintTypeName VARCHAR(50) UNIQUE)
    ''')

    # Create a table named 'Constraints'
    cursor.execute('''
    CREATE TABLE Constraints (
        ConstraintID INTEGER PRIMARY KEY,
        ProjectID INTEGER,
        InteractionID INTEGER,
        ConstraintTypeID INTEGER,
        Description  VARCHAR,
        Severity VARCHAR(10),
        FOREIGN KEY (ProjectID) REFERENCES Project(ProjectID),
        FOREIGN KEY (InteractionID) REFERENCES InteractionLog(InteractionID),
        FOREIGN KEY (ConstraintTypeID) REFERENCES ConstraintType(ConstraintTypeID)
    )
    ''')

    # Create a lookup table named 'Requirements'
    cursor.execute('''
    CREATE TABLE RequirementCategories (
        RequirementCategoryID INTEGER PRIMARY KEY,
        RequirementCategoryName VARCHAR(75) UNIQUE)
    ''')

    # Create a table named 'Requirements'
    cursor.execute('''
    CREATE TABLE Requirements (
        RequirementID INTEGER PRIMARY KEY,
        ProjectID INTEGER,
        InteractionID INTEGER,
        Type BOOLEAN,
        Description VARCHAR,
        Status VARCHAR(10),
        PriorityType VARCHAR(20),
        RequirementCategoryID INTEGER,
        FOREIGN KEY (ProjectID) REFERENCES Project(ProjectID),
        FOREIGN KEY (InteractionID) REFERENCES InteractionLog(InteractionID),
        FOREIGN KEY (RequirementCategoryID) REFERENCES RequirementCategories(RequirementCategoryID)
    )
    ''')

    # Insert some example data into the industry table
    industries = [
        ('Aerospace & Defense',),
        ('Automotive',),
        ('Banking',),
        ('Communications, Media & Technology',),
        ('Education',),
        ('Healthcare',),
        ('Information Services',),
        ('Insurance',),
        ('Manufacturing',),
        ('Oil & Gas',),
        ('Blue Economy',),
        ('Capital Markets',),
        ('Consumer Goods',),
        ('Life Sciences',)
    ]
    cursor.executemany('INSERT INTO industry (IndustryName) VALUES (?)', industries)

    # Insert example data into the SourceType table
    source_types = [
        ('Email',),
        ('Meeting Transcript',),
        ('Chat',),
        ('Document',)
    ]
    cursor.executemany('INSERT INTO SourceType (SourceTypeName) VALUES (?)', source_types)

    # Insert example data into the ConstraintType table
    constraint_types = [
        ('Timeframe',),
        ('Budget',),
        ('Compliance',),
        ('Tech limitations',),
        ('Risk',),
        ('Resources',)
    ]
    cursor.executemany('INSERT INTO ConstraintType (ConstraintTypeName) VALUES (?)', constraint_types)

    conn.commit()
    conn.close()


def check_missing_fields(tabel_name,data):

  # gets column from specific table
  query = f"PRAGMA table_info({tabel_name})"
  cursor.execute(query)
  columns = cursor.fetchall()
  essential_fields={}

  try:
    for EachColumn in range(1,len(columns)):
      column = columns[EachColumn]

      essential_fields[column[1]] = data[column[1]]


    missing_fields = [key for key, value in essential_fields.items() if value == ""]
    if missing_fields:
      raise KeyError(f"{', '.join(missing_fields)}")

    return essential_fields

  except KeyError as e:
    return(f"Error: Missing key - {e}")
  except Exception as e:
    return(f"Unexpected error: {e}")

def add_client(data):

  essential_fields = check_missing_fields('Clients',data)
  if isinstance(essential_fields, str):
    return essential_fields

  cursor.execute("SELECT IndustryID FROM Industry WHERE IndustryName = ?", (essential_fields['IndustryID'],))
  result = cursor.fetchone()

  if result:
        essential_fields['IndustryID'] = result[0]
  else:
    cursor.execute("INSERT INTO Industry (IndustryName) VALUES (?)", (essential_fields['IndustryID'],))
    conn.commit()
    essential_fields['IndustryID'] = cursor.lastrowid

  try:
    cursor.execute("INSERT INTO Clients (ClientName, ContactEmail, ContactNumber, Location, IndustryID) VALUES (?, ?, ?, ?, ?)",
                  (essential_fields['ClientName'], essential_fields['ContactEmail'], essential_fields['ContactNumber'],
                    essential_fields['Location'], essential_fields['IndustryID']))
    conn.commit()
    ClientID = cursor.lastrowid
    return ClientID

  except sqlite3.Error as e:
    if str(e) == "UNIQUE constraint failed: Clients.ClientName":
      cursor.execute("SELECT ClientID FROM Clients WHERE ClientName = ?", (essential_fields['ClientName'],))
      result = cursor.fetchone()
      ClientID = result[0]
      return ClientID
    else:
      return(f"Error: {e}")

def add_project(data, Client_ID):
  data['ClientID'] = Client_ID
  essential_fields = check_missing_fields('Project',data)
  if isinstance(essential_fields, str):
    return essential_fields

  try:
    if isinstance(essential_fields['StartDate'], str):
      essential_fields['StartDate'] = datetime.strptime(essential_fields['StartDate'], "%Y-%m-%d").date()
  except ValueError:
    return(f"Error: Invalid date format for StartDate.")

  try:
    if isinstance(essential_fields['EndDate'], str):
      essential_fields['EndDate'] = datetime.strptime(essential_fields['EndDate'], "%Y-%m-%d").date()
  except ValueError:
    return(f"Error: Invalid date format for EndDate.")

  try:
    if isinstance(essential_fields['Budget'], str):
      essential_fields['Budget'] = int(essential_fields['Budget'])
  except ValueError:
    return(f"Error: Invalid value format for Budget. Use numbers and not text")

  try:
    if isinstance(essential_fields['NumUsers'], str):
      essential_fields['NumUsers'] = int(essential_fields['NumUsers'])
  except ValueError:
    return(f"Error: Invalid value format for NumUsers. Use numbers and not text")



  try:
    cursor.execute("INSERT INTO Project (ProjectName, StartDate, EndDate, NumUsers, ProjectStatus, Budget, DeliveryModel, ClientID) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  (essential_fields['ProjectName'], essential_fields['StartDate'], essential_fields['EndDate'],
                    essential_fields['NumUsers'], essential_fields['ProjectStatus'], essential_fields['Budget'],
                    essential_fields['DeliveryModel'], essential_fields['ClientID']))
    conn.commit()
    ProjectID = cursor.lastrowid
    return ProjectID
  except sqlite3.Error as e:
    return(f"Error: {e}")
  
def add_project_technology(data,ProjectID):
    for each in data:
      essential_fields = check_missing_fields('TechnologyStack',each)
      if isinstance(essential_fields, str):
        msg = essential_fields+" in TechnologyStack "+str(data.index(each))
        return msg

      cursor.execute("SELECT TechID FROM TechnologyStack WHERE TechName = ?", (essential_fields['TechName'],))
      result = cursor.fetchone()

      if result:
            essential_fields['TechName'] = result[0]
      else:
        cursor.execute("INSERT INTO TechnologyStack (TechName,Category) VALUES (?,?)",
         (essential_fields['TechName'],essential_fields['Category']))
        conn.commit()
        essential_fields['TechName'] = cursor.lastrowid

      try:
          cursor.execute("INSERT INTO ProjectTechnology (ProjectID, TechID, Status) VALUES (?, ?, ?)",
                        (ProjectID, essential_fields['TechName'], each['Status']))
          conn.commit()

      except sqlite3.Error as e:
        if str(e) == "UNIQUE constraint failed: ProjectTechnology.ProjectID, ProjectTechnology.TechID":
          pass
        else:
          return(f"Error: {e}")
    return True

def add_Interaction_Log(data):

  essential_fields = check_missing_fields('InteractionLog',data)
  if isinstance(essential_fields, str):
    return essential_fields
  #
  cursor.execute("SELECT SourceTypeID FROM SourceType WHERE SourceTypeName = ?", (essential_fields['SourceTypeID'],))
  result = cursor.fetchone()

  #
  if result:
      essential_fields['SourceTypeID'] = result[0]
  else:
    cursor.execute("INSERT INTO SourceType (SourceTypeName) VALUES (?)", (essential_fields['SourceTypeID'],))
    conn.commit()
    essential_fields['SourceTypeID'] = cursor.lastrowid

  #
  try:
    if isinstance(essential_fields['Timestamp'], str):
      essential_fields['Timestamp'] = datetime.strptime(essential_fields['Timestamp'], "%Y-%m-%dT%H:%M:%S").date()
  except ValueError:
    return(f"Error: Invalid date format for Timestamp.")


  try:
    cursor.execute("SELECT InteractionID FROM InteractionLog WHERE Timestamp = ? AND SourceTypeID = ? AND RawText = ? AND ExtractedSummary = ?",
                  (essential_fields['Timestamp'], essential_fields['SourceTypeID'], essential_fields['RawText'],
                    essential_fields['ExtractedSummary']))
    result = cursor.fetchone()

    if result:
      InteractionID = result[0]
      return InteractionID
    else:
      cursor.execute("INSERT INTO InteractionLog (Timestamp, SourceTypeID, RawText, ExtractedSummary) VALUES (?, ?, ?, ?)",
                    (essential_fields['Timestamp'], essential_fields['SourceTypeID'], essential_fields['RawText'],
                      essential_fields['ExtractedSummary']))
    conn.commit()
    InteractionID = cursor.lastrowid
    return InteractionID

  except sqlite3.Error as e:
    return(f"Error: {e}")

def add_Requirements(data,ProjectID):
  for each in data:
    each['ProjectID'] = ProjectID
    essential_fields = check_missing_fields('Requirements',each)
    if isinstance(essential_fields, str):
      msg = essential_fields+" in Requirements "+str(data.index(each))
      return msg

    cursor.execute("SELECT RequirementCategoryID FROM RequirementCategories WHERE RequirementCategoryName = ?",
                   (essential_fields['RequirementCategoryID'],))
    result = cursor.fetchone()

    if result:
      essential_fields['RequirementCategoryID'] = result[0]
    else:
      cursor.execute("INSERT INTO RequirementCategories (RequirementCategoryName) VALUES (?)",
                     (essential_fields['RequirementCategoryID'],))
      conn.commit()
      essential_fields['RequirementCategoryID'] = cursor.lastrowid

    SourceID = add_Interaction_Log(essential_fields["InteractionID"])
    if isinstance(SourceID, str):
      return SourceID
    else:
      essential_fields['InteractionID'] = SourceID


    try:
      if isinstance(essential_fields['Type'], str):
        if essential_fields['Type'] == "Functional":
          essential_fields['Type'] = 1
        elif essential_fields['Type'] == "Non-functional":
          essential_fields['Type'] = 0
        else:
          raise ValueError("Error: Invalid value for Type. Type can only be Functional or Non-functional ")
      else:
        raise ValueError("Error: Invalid value for Type. Type can only be Functional or Non-functional ")

    except ValueError as e:
      return(f"Error: {e}")

    try:
      cursor.execute("SELECT RequirementID FROM Requirements WHERE ProjectID = ? AND InteractionID = ? AND Type = ? AND Description = ? AND Status = ? AND PriorityType = ? AND RequirementCategoryID = ?",
                    (essential_fields['ProjectID'], essential_fields['InteractionID'], essential_fields['Type'],
                      essential_fields['Description'], essential_fields['Status'], essential_fields['PriorityType'],
                      essential_fields['RequirementCategoryID']))
      result = cursor.fetchone()

      if result:
        print("Requirements details already in database. \n")
      else:
        cursor.execute("INSERT INTO Requirements (ProjectID, InteractionID, Type, Description, Status, PriorityType, RequirementCategoryID) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (essential_fields['ProjectID'], essential_fields['InteractionID'], essential_fields['Type'],
                          essential_fields['Description'], essential_fields['Status'], essential_fields['PriorityType'],
                          essential_fields['RequirementCategoryID']))
        conn.commit()

    except sqlite3.Error as e:
      return(f"Error: {e}")
  return True

def add_Constraints(data,ProjectID):
  for each in data:
    each['ProjectID'] = ProjectID
    essential_fields = check_missing_fields('Constraints',each)
    if isinstance(essential_fields, str):
      msg = essential_fields+" in Constraints "+str(data.index(each))
      return msg

    cursor.execute("SELECT ConstraintTypeID FROM ConstraintType WHERE ConstraintTypeName = ?",
                   (essential_fields['ConstraintTypeID'],))
    result = cursor.fetchone()

    if result:
      essential_fields['ConstraintTypeID'] = result[0]
    else:
      cursor.execute("INSERT INTO ConstraintType (ConstraintTypeName) VALUES (?)",
                     (essential_fields['ConstraintTypeID'],))
      conn.commit()
      essential_fields['ConstraintTypeID'] = cursor.lastrowid

    SourceID = add_Interaction_Log(essential_fields["InteractionID"])
    if isinstance(SourceID, str):
      return SourceID
    else:
      essential_fields['InteractionID'] = SourceID



    try:
      cursor.execute("SELECT ConstraintID FROM Constraints WHERE ProjectID = ? AND InteractionID = ? AND ConstraintTypeID = ? AND Description = ? AND Severity = ?",
                    (essential_fields['ProjectID'], essential_fields['InteractionID'], essential_fields['ConstraintTypeID'],
                      essential_fields['Description'], essential_fields['Severity']))
      result = cursor.fetchone()

      if result:
        print("Constraint details already in database. \n")
      else:
        cursor.execute("INSERT INTO Constraints (ProjectID, InteractionID, ConstraintTypeID, Description, Severity) VALUES (?, ?, ?, ?, ?)",
                        (essential_fields['ProjectID'], essential_fields['InteractionID'], essential_fields['ConstraintTypeID'],
                          essential_fields['Description'], essential_fields['Severity']))
        conn.commit()

    except sqlite3.Error as e:
      return(f"Error: {e}")
  return True

def main(data: dict):

  global conn, cursor
  # Connect to the SQLite database
  database_path = 'my_DB.db'
  conn = sqlite3.connect(database_path)
  cursor = conn.cursor()

  ClientID = add_client(data['Clients'])
  if isinstance(ClientID, str):
    return ClientID
  
  ProjectID = add_project(data['Project'], ClientID)
  if isinstance(ProjectID, str):
    return ProjectID
  
  tech_stack = add_project_technology(data['ProjectTechnology'], ProjectID)
  if isinstance(tech_stack, str):
    return tech_stack
  
  requirements = add_Requirements(data['Requirements'], ProjectID)
  if isinstance(requirements, str):
    return requirements       
  
  constraints = add_Constraints(data['Constraints'], ProjectID)
  if isinstance(constraints, str):
    return constraints    
    
  conn.close()
  return True




if __name__ == "__main__":
    create_database()
    input_file = 'test.json'
    with open(input_file, 'r') as file:
        input = json.load(file)
    print(main(input))

