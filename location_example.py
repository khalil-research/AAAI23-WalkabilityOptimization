from docplex.cp.model import CpoModel
from collections import namedtuple

# -----------------------------------------------------------------------------
# Initialize the problem data
# -----------------------------------------------------------------------------

Warehouse = namedtuple('Warehouse', ('city',  # Name of the city
                                     'capacity',  # Capacity of the warehouse
                                     'cost',  # Warehouse building cost
                                     ))

# List of warehouses
WAREHOUSES = (Warehouse("Bonn", 3, 480),
              Warehouse("Bordeaux", 1, 200),
              Warehouse("London", 2, 320),
              Warehouse("Paris", 4, 340),
              Warehouse("Rome", 1, 300))
NB_WAREHOUSES = len(WAREHOUSES)

# Number of stores
NB_STORES = 8

# Supply cost for each store and warehouse
SUPPLY_COST = ((24, 74, 31, 51, 84),
               (57, 54, 86, 61, 68),
               (57, 67, 29, 91, 71),
               (54, 54, 65, 82, 94),
               (98, 81, 16, 61, 27),
               (13, 92, 34, 94, 87),
               (54, 72, 41, 12, 78),
               (54, 64, 65, 89, 89))

# -----------------------------------------------------------------------------
# Build the model
# -----------------------------------------------------------------------------

# Create CPO model
mdl = CpoModel()

# Create one variable per store to contain the index of its supplying warehouse
NB_WAREHOUSES = len(WAREHOUSES)
supplier = mdl.integer_var_list(NB_STORES, 0, NB_WAREHOUSES - 1, "supplier")

# Create one variable per warehouse to indicate if it is open (1) or not (0)
open = mdl.integer_var_list(NB_WAREHOUSES, 0, 1, "open")
f={}
tst=4000
for n in range(tst):
    f[n] = mdl.float_var(name=f'f[{n}]')
ints={}
for n in range(tst):
    ints[n] = mdl.integer_var(name=f'ints[{n}]')
f2 = {}
for n in range(tst):
    f2[n] = mdl.float_var(name=f'f2[{n}]')

for s in range(tst):
    mdl.add(f[n]==900)
# Add constraints stating that the supplying warehouse of each store must be open
for s in supplier:
    mdl.add(mdl.element(open, s) == 1)

# Add constraints stating that the number of stores supplied by each warehouse must not exceed its capacity
for wx in range(NB_WAREHOUSES):
    mdl.add(mdl.count(supplier, wx) <= WAREHOUSES[wx].capacity)

# Build an expression that computes total cost
total_cost = mdl.scal_prod(open, [w.cost for w in WAREHOUSES])
for sx in range(NB_STORES):
    total_cost = total_cost + mdl.element(supplier[sx], SUPPLY_COST[sx])

# Minimize total cost
mdl.add(mdl.minimize(total_cost))

# -----------------------------------------------------------------------------
# Solve the model and display the result
# -----------------------------------------------------------------------------

# Solve model
print("\nSolving model....")
msol = mdl.solve(TimeLimit=10)

# Print solution
if msol:
    for wx in range(NB_WAREHOUSES):
        if msol[open[wx]] == 1:
            print("Warehouse '{}' open to supply stores: {}"
                  .format(WAREHOUSES[wx].city,
                          ", ".join(str(sx) for sx in range(NB_STORES) if msol[supplier[sx]] == wx)))
    print("Total cost is: {}".format(msol.get_objective_values()[0]))
else:
    print("No solution found.")