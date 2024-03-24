FROM mcr.microsoft.com/dotnet/runtime:8.0 AS base
USER $APP_UID
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
ARG BUILD_CONFIGURATION=Release
WORKDIR /src
COPY ["SharpEmbed.Cli/SharpEmbed.Cli.csproj", "SharpEmbed.Cli/"]
RUN dotnet restore "SharpEmbed.Cli/SharpEmbed.Cli.csproj"
COPY . .
WORKDIR "/src/SharpEmbed.Cli"
RUN dotnet build "SharpEmbed.Cli.csproj" -c $BUILD_CONFIGURATION -o /app/build

FROM build AS publish
ARG BUILD_CONFIGURATION=Release
RUN dotnet publish "SharpEmbed.Cli.csproj" -c $BUILD_CONFIGURATION -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "SharpEmbed.Cli.dll"]
