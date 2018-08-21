procedure Test is
   X : Integer;
   Y : Integer;
   B : Boolean;

   procedure F is
   begin
      X := 12;
   end F;

   procedure G is
   begin
      X := 14;
      Y := 42;
   end G;

   P : access procedure;
begin
   if B then
      P := F'Access;
   else
      P := G'Access;
   end if;
end Test;
