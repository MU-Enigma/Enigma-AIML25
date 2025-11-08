import math


def magnitude(vector):
    return math.sqrt(sum([x**2] for x in vector))


def dot(v1, v2, angle):
    return magnitude(v1) * magnitude(v2) * math.cos(angle)


def cosine_similarity(v1, v2, angle):
    return (magnitude(v1) * magnitude(v2) * math.cos(angle)) / (
        magnitude(v1) * magnitude(v2)
    )
