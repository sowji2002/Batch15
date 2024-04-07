FROM dpage/pgadmin4:latest

# Set environment variables for pgAdmin 4
ENV PGADMIN_DEFAULT_EMAIL=user-name@domain-name.com
ENV PGADMIN_DEFAULT_PASSWORD=strong-password

# Expose the pgAdmin 4 port
EXPOSE 80

# Create a volume for pgAdmin data
VOLUME /var/lib/pgadmin

# (Optional) You can add additional commands or configurations here if needed
# CMD ["entrypoint.sh"]