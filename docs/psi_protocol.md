# Diagram
+--------------+                        +------------+    Doubly-Blinded    +--------------+                +-------------+
|              |                        |            |    Client Data       |              |                |   Ordered_  |
|    Client    |   --------------->     |   Server   |   --------------->   |    Client    | -------------> |   Id_List.  |
|              |    Single-Blinded      |            |    Single-Blinded    |              |  Intersection  |   json      |
+--------------+    Client Data         +------------+    Server Data       +--------------+                +-------------+
Blinds own data                     Blinds both data sets                   Blinds server data        Holds intersection for use later
                                                                         Finds data intersection

# Explanation

The client connects to the server after blinding it's own data. This data is then sent to the server, and the server blinds the single-blinded client data alongside it's own data. This data is sent back to the client, who blinds the single-blinded server data to have two sets of doubly-blinded data. It then compares both of these sets, looking to find any overlap between the two. The overlap is then written to a file for future use.