from sage.all import *
from sage.geometry.lattice_polytope import LatticePolytope
import multiprocessing
import csv
import time
import os
from itertools import permutations, product

# Define sets
A01 = Set([(0, 5), (0, 6)])
A02 = Set([(0, -5), (0, -6)])
S01 = Set(A01.subsets(1)).union(Set(A01.subsets(0)))
S02 = Set(A02.subsets(1)).union(Set(A02.subsets(0)))

A11 = Set([(1, 5), (1, 6)])
A12 = Set([(1, -5), (1, -6)])
S11 = Set(A11.subsets(1)).union(Set(A11.subsets(0)))
S12 = Set(A12.subsets(1)).union(Set(A12.subsets(0)))

A21 = Set([(2, 5), (2, 6)])
A22 = Set([(2, -5), (2, -6)])
S21 = Set(A21.subsets(1)).union(Set(A21.subsets(0)))
S22 = Set(A22.subsets(1)).union(Set(A22.subsets(0)))

A31 = Set([(3, 5), (3, 6)])
A32 = Set([(3, -5), (3, -6)])
S31 = Set(A31.subsets(1)).union(Set(A31.subsets(0)))
S32 = Set(A32.subsets(1)).union(Set(A32.subsets(0)))

A41 = Set([(4, 5), (4, 6)])
A42 = Set([(4, -5), (4, -6)])
S41 = Set(A41.subsets(1)).union(Set(A41.subsets(0)))
S42 = Set(A42.subsets(1)).union(Set(A42.subsets(0)))

A5 = Set([(5, -6), (5, -5), (5, -4), (5, -3), (5, -2), (5, -1), (5, 0), (5, 1), (5, 2),
          (5, 3), (5, 4), (5, 5), (5, 6)])
S5 = Set(A5.subsets(2)).union(Set(A5.subsets(1))).union(Set(A5.subsets(0)))

A6 = Set([(6, -6), (6, -5), (6, -4), (6, -3), (6, -2), (6, -1), (6, 0), (6, 1), (6, 2),
          (6, 3), (6, 4), (6, 5), (6, 6)])
S6 = Set(A6.subsets(2)).union(Set(A6.subsets(1))).union(Set(A6.subsets(0)))

components = [S01, S02, S11, S12, S21, S22, S31, S32, S41, S42, S5, S6]
P = cartesian_product(components)
total = P.cardinality()
print(f"Total elements: {total}")

# Generate 2D unimodular transformation matrices
def generate_matrices(n):
    for signs in product([-1, 1], repeat=n):
        for perm in permutations(range(n)):
            mat = Matrix(ZZ, n)
            for i in range(n):
                mat[i, perm[i]] = signs[i]
            yield mat

TRANSFORM_MATRICES = list(generate_matrices(2))

def compute_translate(points):
    vertices = [vector(v) for v in points]
    centroid = sum(vertices) / len(vertices)
    return [tuple(v - centroid) for v in vertices]

def deduplicate(polys):
    unique = []
    seen = set()
    for poly in polys:
        translated = frozenset(compute_translate(poly))
        if translated in seen:
            continue
        unique.append(poly)
        for L in TRANSFORM_MATRICES:
            transformed = frozenset(tuple(L * vector(v)) for v in translated)
            seen.add(transformed)
    return unique

def to_set(p):
    S0 = Set([])
    for i in range(0, 5):
        S1 = p[2 * i]
        S2 = p[2 * i + 1]
        ub = list(S1)[0][1] if S1.cardinality() != 0 else 4
        lb = list(S2)[0][1] if S2.cardinality() != 0 else -4
        temp = Set(cartesian_product([Set([i]), Set(range(lb, ub + 1))]))
        S0 = S0.union(temp)
    for i in range(10, 12):
        S1 = p[i]
        if S1.cardinality() == 1:
            S0 = S0.union(S1)
        elif S1.cardinality() == 2:
            point1, point2 = list(S1)
            lb, ub = sorted([point1[1], point2[1]])
            temp = Set(cartesian_product([Set([i - 5]), Set(range(lb, ub + 1))]))
            S0 = S0.union(temp)
    return S0

def ref_set(S):
    return Set([(-p[0], -p[1]) for p in S])

def is_y_convex(S):
    for i in range(-6, 7):
        x_coords = [p[0] for p in S if p[1] == i]
        if len(x_coords) >= 2:
            lb, ub = min(x_coords), max(x_coords)
            if ub - lb + 1 > len(x_coords):
                return False
    return True

def process_range(start, end):
    valid_sets = []
    valid_count = 0
    processed_count = 0
    total_count = end - start
    last_time = time.time()
    for j in range(start, end):
        p = P.unrank(j)
        S0 = to_set(p).union(ref_set(to_set(p)))
        processed_count += 1
    
        if is_y_convex(S0):
           convex_vertices = lattice_polytope.convex_hull(S0)
           poly = LatticePolytope(convex_vertices)
           if S0.cardinality() == poly.npoints():
              vertex_set = frozenset(tuple(v) for v in convex_vertices)
              valid_sets.append(vertex_set)
              valid_count += 1
              continue
        if processed_count % 100000 == 0 or processed_count == total_count:
            current_time = time.time()
            elapsed = current_time - last_time
            last_time = current_time
            percentage = float(processed_count) / total_count * 100  # Ensure percentage is a float
            print(f"[{multiprocessing.current_process().name}] Progress: {percentage:.2f}%, Found {int(valid_count)} convex sets, Time elapsed: {float(elapsed):.2f}s")
    return (valid_count, processed_count, valid_sets)

if __name__ == '__main__':
    num_processes = 10
    size = total // num_processes
    ranges = [(i*size, (i+1)*size if i != num_processes-1 else total) for i in range(num_processes)]

    with multiprocessing.Pool(num_processes) as pool:
         async_result = pool.starmap_async(process_range, ranges)
         results = async_result.get()

    total_valid = 0
    total_processed = 0
    all_valid = []
    for valid_count, processed_count, valid_sets in results:
        total_valid += valid_count
        total_processed += processed_count
        all_valid.extend(valid_sets)

    # Deduplicate
    print("Starting deduplication...")
    unique_polys = deduplicate(all_valid)
    print(f"Deduplication completed, {len(unique_polys)} unique convex sets remaining")

    # Write results
    with open('unique_final_result.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for poly in unique_polys:
            writer.writerows([list(p) for p in poly])

    print(f"Processing completed! Checked {total_processed} sets, found {total_valid} convex sets, final {len(unique_polys)} unique solutions")