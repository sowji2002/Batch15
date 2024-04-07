# Use the official PostgreSQL image as the base image
FROM postgres:latest

# Set environment variables for PostgreSQL
ENV POSTGRES_USER=user-name
ENV POSTGRES_PASSWORD=strong-password
ENV POSTGRES_DB=postgres

# Expose the PostgreSQL port
EXPOSE 5432

# Create a volume for PostgreSQL data
VOLUME /var/lib/postgresql/data

CMD ["postgres"]