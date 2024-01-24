FROM mcr.microsoft.com/dotnet/runtime:8.0 AS base
USER $APP_UID
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
ARG BUILD_CONFIGURATION=Release
WORKDIR /src
COPY ["SharpEmbed.Console/SharpEmbed.Console.csproj", "SharpEmbed.Console/"]
RUN dotnet restore "SharpEmbed.Console/SharpEmbed.Console.csproj"
COPY . .
WORKDIR "/src/SharpEmbed.Console"
RUN dotnet build "SharpEmbed.Console.csproj" -c $BUILD_CONFIGURATION -o /app/build

FROM build AS publish
ARG BUILD_CONFIGURATION=Release
RUN dotnet publish "SharpEmbed.Console.csproj" -c $BUILD_CONFIGURATION -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "SharpEmbed.Console.dll"]
