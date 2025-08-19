side_groups={
    'A':[-1,1],
    'C':[-1,-1],
    'G':[1,-1],
    'T':[1,1]
}
structure={
    'A':[-1,1],
    'C':[1,-1],
    'G':[-1,-1],
    'T':[1,1]
}
bonds={
    'A':[-1,1],
    'C':[1,1],
    'G':[1,-1],
    'T':[-1,-1]
}

def CGRepresentation(seq:str,encodings:dict)->list:
    """
    Generating Chaos Game Representations (point coordinates) for DNA sequences.

    Parameters:
    - seq: DNA sequence with datatype of str
    - encodings: dictionary type variable that contains values for all of the nucleotides

    Returns:
    - list of coordinates
    """
    seq=seq.replace("\n",'').replace('N','')
    coordinates=[[0,0]]
    for nuc in seq:
        corner=encodings[nuc]
        current=coordinates[-1]
        x=(corner[0]+current[0])/2
        y=(corner[1]+current[1])/2
        coordinates.append([x,y])
    return coordinates[1:]

def FrequencyCGR(coordinates, resolution):
    """
    Generate a Frequency Chaos Game Representation (FCGR) matrix.

    Parameters:
    - coordinates: list of tuples (x, y) representing points in the unit square [0, 1] x [0, 1]
    - resolution: int, the number of bins along each axis (e.g., 256 for a 256x256 matrix)

    Returns:
    - fcgr_matrix: 2D numpy array of shape (resolution, resolution)
    """
    fcgr_matrix = np.zeros((resolution, resolution), dtype=int)

    for x, y in coordinates:
        i = int(x * resolution)
        j = int(y * resolution)

        # Ensure indices are within bounds
        i = min(i, resolution - 1)
        j = min(j, resolution - 1)

        fcgr_matrix[i, j] += 1

    return fcgr_matrix.flatten()
